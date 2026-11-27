// Copyright (c) Meta Platforms, Inc. and affiliates.
// This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

//! Pure-Rust SentencePiece (BPE) tokenizer.
//!
//! The Python reference wraps `sentencepiece.SentencePieceProcessor`. This module instead reads the
//! `tokenizer.model` protobuf directly and implements SentencePiece's BPE segmentation (greedy merge
//! of the adjacent pair with the highest piece score, leftmost on ties) with byte fallback.
//! Llama's model uses an identity normalizer, so no charsmap is needed; models that carry a
//! non-empty `precompiled_charsmap` are rejected rather than silently mis-tokenized.

use anyhow::{bail, ensure, Result};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

const SPACE: char = '\u{2581}'; // "▁"

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl Kind {
    fn from_proto(v: u64) -> Result<Kind> {
        Ok(match v {
            1 => Kind::Normal,
            2 => Kind::Unknown,
            3 => Kind::Control,
            4 => Kind::UserDefined,
            5 => Kind::Unused,
            6 => Kind::Byte,
            other => bail!("unknown SentencePiece piece type {other}"),
        })
    }
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    score: f32,
    kind: Kind,
}

/// SentencePiece (BPE) tokenizer read straight from a `tokenizer.model` protobuf.
#[derive(Debug, Clone)]
pub struct SpmTokenizer {
    pieces: Vec<Piece>,
    /// Pieces that BPE merges may produce (normal + user-defined).
    lookup: HashMap<String, i64>,
    /// Every piece text (including control pieces such as `<s>`), for `token_to_id`.
    all: HashMap<String, i64>,
    byte_ids: [i64; 256],
    pub n_words: usize,
    pub bos_id: i64,
    pub eos_id: i64,
    pub pad_id: i64,
    pub unk_id: i64,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

// ---------------------------------------------------------------------------------------------
// Minimal protobuf reader
// ---------------------------------------------------------------------------------------------

enum Wire<'a> {
    Varint(u64),
    Fixed64,
    Len(&'a [u8]),
    Fixed32(u32),
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..70).step_by(7) {
            let Some(&b) = self.buf.get(self.pos) else {
                bail!("truncated protobuf varint");
            };
            self.pos += 1;
            if shift < 64 {
                v |= u64::from(b & 0x7f) << shift;
            }
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("protobuf varint too long")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.buf.len(), "truncated protobuf field");
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn next(&mut self) -> Result<Option<(u32, Wire<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        let wire = match key & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                Wire::Len(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                Wire::Fixed32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            other => bail!("unsupported protobuf wire type {other}"),
        };
        Ok(Some((field, wire)))
    }
}

fn parse_byte_piece(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

// ---------------------------------------------------------------------------------------------
// BPE agenda
// ---------------------------------------------------------------------------------------------

struct Sym {
    start: usize,
    end: usize,
    prev: isize,
    next: isize,
    alive: bool,
}

struct Cand {
    score: f32,
    left: usize,
    right: usize,
    /// Byte length of the merged text when queued; used to detect stale entries.
    len: usize,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    // Max-heap: higher score first; on equal score the leftmost pair wins.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl SpmTokenizer {
    /// Initializes the tokenizer with a SentencePiece model file.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        ensure!(
            path.is_file(),
            "tokenizer model not found: {}",
            path.display()
        );
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parses a serialized SentencePiece `ModelProto`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let (mut unk_id, mut bos_id, mut eos_id, mut pad_id) = (0i32, 1i32, 2i32, -1i32);
        let (mut add_dummy_prefix, mut remove_extra_whitespaces, mut escape_whitespaces) =
            (true, true, true);

        let mut r = Reader::new(bytes);
        while let Some((field, wire)) = r.next()? {
            match (field, wire) {
                (1, Wire::Len(b)) => {
                    let (mut text, mut score, mut kind) = (String::new(), 0f32, Kind::Normal);
                    let mut pr = Reader::new(b);
                    while let Some((f, w)) = pr.next()? {
                        match (f, w) {
                            (1, Wire::Len(s)) => text = String::from_utf8_lossy(s).into_owned(),
                            (2, Wire::Fixed32(v)) => score = f32::from_bits(v),
                            (3, Wire::Varint(v)) => kind = Kind::from_proto(v)?,
                            _ => {}
                        }
                    }
                    pieces.push(Piece { text, score, kind });
                }
                (2, Wire::Len(b)) => {
                    let mut tr = Reader::new(b);
                    while let Some((f, w)) = tr.next()? {
                        // int32 fields: negative values arrive sign-extended to 64 bits.
                        if let Wire::Varint(v) = w {
                            match f {
                                40 => unk_id = v as i64 as i32,
                                41 => bos_id = v as i64 as i32,
                                42 => eos_id = v as i64 as i32,
                                43 => pad_id = v as i64 as i32,
                                _ => {}
                            }
                        }
                    }
                }
                (3, Wire::Len(b)) => {
                    let mut nr = Reader::new(b);
                    while let Some((f, w)) = nr.next()? {
                        match (f, w) {
                            (2, Wire::Len(cm)) if !cm.is_empty() => bail!(
                                "this SentencePiece model has a non-empty normalizer charsmap, \
                                 which this tokenizer does not implement (Llama models use none)"
                            ),
                            (3, Wire::Varint(v)) => add_dummy_prefix = v != 0,
                            (4, Wire::Varint(v)) => remove_extra_whitespaces = v != 0,
                            (5, Wire::Varint(v)) => escape_whitespaces = v != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        ensure!(!pieces.is_empty(), "SentencePiece model contains no pieces");

        let mut lookup = HashMap::new();
        let mut all = HashMap::new();
        let mut byte_ids = [-1i64; 256];
        for (i, p) in pieces.iter().enumerate() {
            all.entry(p.text.clone()).or_insert(i as i64);
            match p.kind {
                Kind::Normal | Kind::UserDefined => {
                    lookup.entry(p.text.clone()).or_insert(i as i64);
                }
                Kind::Byte => {
                    if let Some(b) = parse_byte_piece(&p.text) {
                        byte_ids[b as usize] = i as i64;
                    }
                }
                _ => {}
            }
        }

        Ok(SpmTokenizer {
            n_words: pieces.len(),
            pieces,
            lookup,
            all,
            byte_ids,
            bos_id: i64::from(bos_id),
            eos_id: i64::from(eos_id),
            pad_id: i64::from(pad_id),
            unk_id: i64::from(unk_id),
            add_dummy_prefix,
            remove_extra_whitespaces,
            escape_whitespaces,
        })
    }

    /// Encodes a string into a list of token IDs, optionally with BOS / EOS.
    pub fn encode_i64(&self, s: &str, bos: bool, eos: bool) -> Vec<i64> {
        let mut t = Vec::new();
        if bos {
            t.push(self.bos_id);
        }
        t.extend(self.encode_pieces(s));
        if eos {
            t.push(self.eos_id);
        }
        t
    }


    /// `u32` ids, as used by the rest of this crate. A model without a usable `<unk>` falls back
    /// to id 0 for text it cannot represent (Llama models always have byte pieces, so this is
    /// not reached for them).
    pub fn encode(&self, s: &str, add_bos: bool) -> Vec<u32> {
        self.encode_i64(s, add_bos, false)
            .into_iter()
            .map(|i| u32::try_from(i).unwrap_or(0))
            .collect()
    }

    /// Decodes ids. `skip_special` drops control pieces (`<s>`, `</s>`); with `false` they are
    /// rendered by name. Out-of-range ids are ignored.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        if skip_special {
            let v: Vec<i64> = ids.iter().map(|&i| i64::from(i)).collect();
            return self.decode_i64(&v);
        }
        // Keep control pieces: decode runs between them, splice their text in.
        let mut out = String::new();
        let mut run: Vec<i64> = Vec::new();
        for &id in ids {
            match self.pieces.get(id as usize) {
                Some(p) if p.kind == Kind::Control => {
                    out.push_str(&self.decode_i64(&run));
                    run.clear();
                    out.push_str(&p.text);
                }
                _ => run.push(i64::from(id)),
            }
        }
        out.push_str(&self.decode_i64(&run));
        out
    }

    /// Id of an exact piece text (normal, control, user-defined, ...), e.g. `"</s>"`.
    pub fn token_to_id(&self, piece: &str) -> Option<u32> {
        self.all.get(piece).and_then(|&i| u32::try_from(i).ok())
    }

    pub fn bos(&self) -> Option<u32> {
        u32::try_from(self.bos_id).ok()
    }

    pub fn eos(&self) -> Option<u32> {
        u32::try_from(self.eos_id).ok()
    }

    pub fn vocab_size(&self) -> usize {
        self.n_words
    }

    /// Decodes a list of token IDs into a string.
    pub fn decode_i64(&self, ids: &[i64]) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        fn flush(out: &mut String, bytes: &mut Vec<u8>) {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        }
        for &id in ids {
            let Some(p) = usize::try_from(id).ok().and_then(|i| self.pieces.get(i)) else {
                continue;
            };
            match p.kind {
                Kind::Byte => match parse_byte_piece(&p.text) {
                    Some(b) => bytes.push(b),
                    None => {
                        flush(&mut out, &mut bytes);
                        out.push_str(&p.text);
                    }
                },
                Kind::Control | Kind::Unused => flush(&mut out, &mut bytes),
                Kind::Unknown => {
                    flush(&mut out, &mut bytes);
                    out.push_str(" \u{2047} ");
                }
                Kind::Normal | Kind::UserDefined => {
                    flush(&mut out, &mut bytes);
                    out.push_str(&p.text);
                }
            }
        }
        flush(&mut out, &mut bytes);

        // Drop the dummy prefix that `encode` added, then un-escape spaces.
        if self.add_dummy_prefix {
            let lead = if self.escape_whitespaces { SPACE } else { ' ' };
            if let Some(rest) = out.strip_prefix(lead) {
                out = rest.to_string();
            }
        }
        if self.escape_whitespaces {
            out = out.replace(SPACE, " ");
        }
        out
    }

    fn normalize(&self, s: &str) -> String {
        let mut text: String = if self.remove_extra_whitespaces {
            s.split(' ')
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            s.to_string()
        };
        if text.is_empty() {
            return text;
        }
        if self.add_dummy_prefix {
            text.insert(0, ' ');
        }
        if self.escape_whitespaces {
            text = text.replace(' ', &SPACE.to_string());
        }
        text
    }

    fn push_pair(
        &self,
        text: &str,
        syms: &[Sym],
        heap: &mut BinaryHeap<Cand>,
        left: isize,
        right: isize,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (left as usize, right as usize);
        let merged = &text[syms[l].start..syms[r].end];
        if let Some(&id) = self.lookup.get(merged) {
            heap.push(Cand {
                score: self.pieces[id as usize].score,
                left: l,
                right: r,
                len: merged.len(),
            });
        }
    }

    fn encode_pieces(&self, s: &str) -> Vec<i64> {
        let text = self.normalize(s);
        if text.is_empty() {
            return Vec::new();
        }

        // One symbol per character.
        let n = text.chars().count();
        let mut syms: Vec<Sym> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, ch))| Sym {
                start,
                end: start + ch.len_utf8(),
                prev: i as isize - 1,
                next: if i + 1 == n { -1 } else { i as isize + 1 },
                alive: true,
            })
            .collect();

        let mut heap = BinaryHeap::new();
        for i in 1..n {
            self.push_pair(&text, &syms, &mut heap, i as isize - 1, i as isize);
        }

        while let Some(c) = heap.pop() {
            let (l, r) = (c.left, c.right);
            if !syms[l].alive || !syms[r].alive {
                continue;
            }
            // Stale if either side changed since the candidate was queued.
            if (syms[l].end - syms[l].start) + (syms[r].end - syms[r].start) != c.len {
                continue;
            }
            syms[l].end = syms[r].end;
            syms[r].alive = false;
            let nr = syms[r].next;
            syms[l].next = nr;
            if nr >= 0 {
                syms[nr as usize].prev = l as isize;
            }
            let (pl, nl) = (syms[l].prev, syms[l].next);
            self.push_pair(&text, &syms, &mut heap, pl, l as isize);
            self.push_pair(&text, &syms, &mut heap, l as isize, nl);
        }

        let mut ids = Vec::new();
        let mut cur = 0isize;
        while cur >= 0 {
            let sym = &syms[cur as usize];
            let piece = &text[sym.start..sym.end];
            match self.lookup.get(piece) {
                Some(&id) => ids.push(id),
                None => self.push_fallback(piece, &mut ids),
            }
            cur = sym.next;
        }
        ids
    }

    /// Byte fallback for text that is not a vocabulary piece; `<unk>` if the model has no byte pieces.
    fn push_fallback(&self, piece: &str, ids: &mut Vec<i64>) {
        if piece.bytes().all(|b| self.byte_ids[b as usize] >= 0) {
            ids.extend(piece.bytes().map(|b| self.byte_ids[b as usize]));
        } else {
            ids.push(self.unk_id);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::tokenizer::Tokenizer;

    pub const SP: &str = "\u{2581}";

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
    }

    fn field_len(field: u32, payload: &[u8], out: &mut Vec<u8>) {
        varint(u64::from(field) << 3 | 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    fn field_varint(field: u32, v: u64, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3, out);
        varint(v, out);
    }

    /// Serialize a SentencePiece `ModelProto` (pieces, trainer ids, normalizer flags).
    pub fn sp_model(pieces: &[(&str, f32, u64)], remove_extra_ws: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for (text, score, kind) in pieces {
            let mut p = Vec::new();
            field_len(1, text.as_bytes(), &mut p);
            varint(2 << 3 | 5, &mut p);
            p.extend_from_slice(&score.to_le_bytes());
            field_varint(3, *kind, &mut p);
            field_len(1, &p, &mut out);
        }
        let mut trainer = Vec::new();
        field_varint(40, 0, &mut trainer); // unk
        field_varint(41, 1, &mut trainer); // bos
        field_varint(42, 2, &mut trainer); // eos
        field_varint(43, (-1i64) as u64, &mut trainer); // pad = -1 (sign-extended)
        field_len(2, &trainer, &mut out);
        let mut norm = Vec::new();
        field_len(1, b"identity", &mut norm);
        field_varint(3, 1, &mut norm); // add_dummy_prefix
        field_varint(4, remove_extra_ws as u64, &mut norm);
        field_varint(5, 1, &mut norm); // escape_whitespaces
        field_len(3, &norm, &mut out);
        out
    }

    /// `<unk> <s> </s>`, 256 byte pieces, then a small BPE vocabulary.
    pub fn tiny_spm_tokenizer() -> Tokenizer {
        Tokenizer::from_spm_bytes(&tiny_spm_bytes()).unwrap()
    }

    pub fn tiny_spm_bytes() -> Vec<u8> {
        let mut pieces: Vec<(String, f32, u64)> =
            vec![("<unk>".into(), 0.0, 2), ("<s>".into(), 0.0, 3), ("</s>".into(), 0.0, 3)];
        for b in 0..=255u32 {
            pieces.push((format!("<0x{b:02X}>"), 0.0, 6));
        }
        let normal: &[(&str, f32)] = &[
            (SP, -10.0), ("a", -9.0), ("b", -8.0), ("c", -7.0), ("[", -6.0), ("]", -6.0),
            ("/", -6.0), ("I", -6.0), ("N", -6.0), ("S", -6.0), ("T", -6.0), ("h", -6.0),
            ("i", -6.0), ("ab", -3.0), ("bc", -2.5), ("abc", -1.0),
        ];
        for (t, s) in normal {
            pieces.push((t.to_string(), *s, 1));
        }
        pieces.push((format!("{SP}a"), -2.0, 1));
        pieces.push((format!("{SP}ab"), -1.5, 1));
        pieces.push((format!("{SP}h"), -2.0, 1));
        pieces.push((format!("{SP}hi"), -1.0, 1));
        let refs: Vec<(&str, f32, u64)> = pieces.iter().map(|(t, s, k)| (t.as_str(), *s, *k)).collect();
        sp_model(&refs, false)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn tiny() -> SpmTokenizer {
        SpmTokenizer::from_bytes(&tiny_spm_bytes()).unwrap()
    }

    fn id_of(tok: &SpmTokenizer, piece: &str) -> u32 {
        let ids = tok.encode(piece, false);
        assert_eq!(ids.len(), 1, "{piece:?} should be one token, got {ids:?}");
        ids[0]
    }

    #[test]
    fn special_ids_and_vocab_size() {
        let t = tiny();
        assert_eq!((t.unk_id, t.bos_id, t.eos_id), (0, 1, 2));
        assert_eq!(t.pad_id, -1, "negative int32 must survive the protobuf varint");
        assert_eq!(t.n_words, 3 + 256 + 16 + 4);
        assert_eq!((t.bos(), t.eos()), (Some(1), Some(2)));
        assert_eq!(t.token_to_id("</s>"), Some(2), "control pieces are addressable by name");
        assert_eq!(t.token_to_id("nope"), None);
    }

    #[test]
    fn bpe_merges_by_score_then_leftmost() {
        let t = tiny();
        // "▁abc": ▁a (-2) merges first, then ▁ab (-1.5); "▁abc" is not a piece => ["▁ab", "c"].
        let ids = t.encode("abc", false);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], id_of(&t, "ab"));
        assert_eq!(t.decode(&[ids[1]], true), "c");
        assert_eq!(t.decode(&ids, true), "abc");
    }

    #[test]
    fn bos_and_empty() {
        let t = tiny();
        assert_eq!(t.encode("hi", true).first(), Some(&1));
        assert_eq!(t.encode("hi", true).len(), 2); // <s> "▁hi"
        assert!(t.encode("", false).is_empty());
        assert_eq!(t.encode("", true), vec![1]);
    }

    #[test]
    fn byte_fallback_and_roundtrip() {
        let t = tiny();
        let ids = t.encode("a\u{e9}", false); // é is not in the vocab -> 2 UTF-8 byte tokens
        assert_eq!(ids.len(), 3);
        assert_eq!(t.decode(&ids, true), "a\u{e9}");
        assert_eq!(t.decode(&ids[1..2], true).chars().next(), Some('\u{FFFD}'));
    }

    #[test]
    fn whitespace_is_preserved_when_not_normalized() {
        let t = tiny();
        for s in ["a  b", "  a", "a ", "ab  c  ", " a"] {
            assert_eq!(t.decode(&t.encode(s, false), true), s, "round trip of {s:?}");
        }
    }

    #[test]
    fn control_pieces_decode_to_nothing_unless_requested() {
        let t = tiny();
        assert_eq!(t.decode(&[1, 2], true), "");
        assert_eq!(t.decode(&[1, 2], false), "<s></s>");
    }

    #[test]
    fn special_text_is_not_parsed_into_control_tokens() {
        // Unlike HF added tokens, "</s>" typed as text is ordinary characters here.
        let t = tiny();
        assert!(!t.encode("</s>", false).contains(&2));
    }

    #[test]
    fn remove_extra_whitespaces_flag_is_honoured() {
        let t = SpmTokenizer::from_bytes(&sp_model(
            &[("<unk>", 0.0, 2), ("<s>", 0.0, 3), ("</s>", 0.0, 3), (SP, -1.0, 1), ("a", -1.0, 1), ("b", -1.0, 1)],
            true,
        ))
        .unwrap();
        assert_eq!(t.decode(&t.encode("  a   b  ", false), true), "a b");
    }

    #[test]
    fn charsmap_models_are_rejected() {
        let mut bytes = sp_model(&[("<unk>", 0.0, 2)], false);
        // normalizer_spec { precompiled_charsmap = "x" }
        bytes.extend_from_slice(&[0x1a, 0x03, 0x12, 0x01, b'x']);
        assert!(SpmTokenizer::from_bytes(&bytes).is_err());
    }

    #[test]
    fn facade_loads_tokenizer_model_from_a_directory_and_streams() {
        let dir = std::env::temp_dir().join(format!("nexgen_llm_spm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.model"), tiny_spm_bytes()).unwrap();
        let tok = crate::tokenizer::load_tokenizer(&dir).unwrap();
        assert!(tok.is_sentencepiece());
        let ids = tok.encode("abc hi", false).unwrap();
        assert_eq!(tok.decode(&ids, true).unwrap(), "abc hi");
        // incremental detokenizer must reproduce the full decode, including the first space
        let mut s = crate::tokenizer::TokenStream::new(&tok);
        let mut out = String::new();
        for &id in &ids {
            if let Some(t) = s.push(id).unwrap() {
                out.push_str(&t);
            }
        }
        if let Some(t) = s.flush().unwrap() {
            out.push_str(&t);
        }
        assert_eq!(out, "abc hi");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
