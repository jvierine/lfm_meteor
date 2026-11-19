use anyhow::{ensure, Context, Result};
use chrono::NaiveDate;
use num_complex::Complex32;
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

pub const HEADER_SIZE: usize = 1024;
pub fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

#[derive(Clone, Debug)]
pub struct Header {
    pub utc_ns: i64,
    pub n: usize,
    pub gate_front_us: f64,
    pub ipp_us: f64,
    pub pulse_us: f64,
    pub az_deg: f64,
    pub el_deg: f64,
    pub rx_az_deg: f64,
    pub rx_el_deg: f64,
}
impl Header {
    pub fn parse(b: &[u8; HEADER_SIZE], offset_hours: i32) -> Result<Self> {
        ensure!(
            u32le(b, 0) == 0xAAAA5555 && u32le(b, 1020) == 0xAA5555AA,
            "invalid frame markers"
        );
        ensure!(
            b[764] == 1,
            "only single-beam DBF data supported; got {}",
            b[764]
        );
        ensure!(
            u16le(b, 36) == 0x1000,
            "requires 4 MHz sampled, 4 MHz LFM waveform; mode={:#x}",
            u16le(b, 36)
        );
        ensure!(b[32] == 0, "requires 430 MHz carrier");
        let n = u32le(b, 768) as usize;
        ensure!(
            (800..=1_000_000).contains(&n),
            "invalid IQ sample count {n}"
        );
        ensure!(u16le(b, 988) == 200, "requires 200 us pulse");
        ensure!(u16le(b, 980) >= 200, "invalid IPP");
        let date = NaiveDate::from_ymd_opt(2000 + b[13] as i32, b[12] as u32, b[15] as u32)
            .context("invalid date")?;
        let time = date
            .and_hms_opt(b[14] as u32, b[17] as u32, b[16] as u32)
            .context("invalid time")?;
        let frac = u16le(b, 18);
        ensure!(frac < 40000, "invalid 25-us fractional second");
        let utc_ns = (time.and_utc().timestamp() - i64::from(offset_hours) * 3600) * 1_000_000_000
            + i64::from(frac) * 25000;
        Ok(Self {
            utc_ns,
            n,
            gate_front_us: u16le(b, 766) as f64,
            ipp_us: u16le(b, 980) as f64,
            pulse_us: 200.0,
            az_deg: u16le(b, 868) as f64 * 360.0 / 65536.0,
            el_deg: u16le(b, 870) as f64 * 360.0 / 65536.0,
            rx_az_deg: u16le(b, 856) as f64 * 360. / 65536.,
            rx_el_deg: u16le(b, 858) as f64 * 360. / 65536.,
        })
    }
}
pub struct Frame {
    pub offset: u64,
    pub header: Header,
    pub header_bytes: [u8; 1024],
    pub iq: Vec<Complex32>,
    pub words: Vec<i32>,
}
pub struct Reader {
    file: BufReader<File>,
    offset_hours: i32,
    trailer: usize,
    pub max_resync_bytes: usize,
    pub recoveries: Vec<(u64, u64)>,
    pub allow_orphaned_tail: bool,
    pub orphaned_tail: Option<(u64, u64)>,
    pub truncated_trailer_missing_bytes: u64,
}
impl Reader {
    pub fn open(path: &Path, offset_hours: i32, trailer: usize) -> Result<Self> {
        Ok(Self {
            file: BufReader::with_capacity(1024 * 1024, File::open(path)?),
            offset_hours,
            trailer,
            max_resync_bytes: 0,
            recoveries: Vec::new(),
            allow_orphaned_tail: false,
            orphaned_tail: None,
            truncated_trailer_missing_bytes: 0,
        })
    }
    pub fn seek(&mut self, offset: u64) -> Result<()> {
        let current = self.file.stream_position()?;
        self.file
            .seek_relative((offset as i128 - current as i128).try_into()?)?;
        Ok(())
    }
    pub fn next(&mut self) -> Result<Option<Frame>> {
        let mut offset = self.file.stream_position()?;
        let mut b = [0u8; 1024];
        // A clean EOF differs from a partial header, which must be reported.
        if self.file.read(&mut b[..1])? == 0 {
            return Ok(None);
        }
        if let Err(e) = self.file.read_exact(&mut b[1..]) {
            if self.allow_orphaned_tail && e.kind() == std::io::ErrorKind::UnexpectedEof {
                return self.finish_tail(offset);
            }
            return Err(e).with_context(|| format!("truncated header at byte {offset}"));
        }
        if (u32le(&b, 0) != 0xAAAA5555 || u32le(&b, 1020) != 0xAA5555AA)
            && self.max_resync_bytes > 0
        {
            self.file.seek(SeekFrom::Start(offset + 4))?;
            let mut window = vec![0u8; self.max_resync_bytes + 1024];
            let mut used = 0;
            while used < window.len() {
                let n = self.file.read(&mut window[used..])?;
                if n == 0 {
                    break;
                }
                used += n;
            }
            let found = (0..used.saturating_sub(1023)).step_by(4).find(|&i| {
                u32le(&window, i) == 0xAAAA5555 && u32le(&window, i + 1020) == 0xAA5555AA
            });
            let i = match found {
                Some(i) => i,
                None if self.allow_orphaned_tail
                    && self.file.stream_position()? == self.file.get_ref().metadata()?.len() =>
                {
                    return self.finish_tail(offset)
                }
                None => anyhow::bail!(
                    "no validated header within configured resync window at byte {offset}"
                ),
            };
            let next = offset + 4 + i as u64;
            ensure!(
                next - offset <= self.max_resync_bytes as u64,
                "recovery exceeds resync limit"
            );
            eprintln!(
                "WARNING: skipped {} bytes at damaged boundary {}",
                next - offset,
                offset
            );
            self.recoveries.push((offset, next - offset));
            offset = next;
            b.copy_from_slice(&window[i..i + 1024]);
            self.file.seek(SeekFrom::Start(offset + 1024))?;
        }
        let header = Header::parse(&b, self.offset_hours)
            .with_context(|| format!("frame at byte {offset}"))?;
        let mut payload = vec![0u8; header.n * 8];
        if let Err(e) = self.file.read_exact(&mut payload) {
            if self.allow_orphaned_tail && e.kind() == std::io::ErrorKind::UnexpectedEof {
                return self.finish_tail(offset);
            }
            return Err(e).with_context(|| format!("truncated IQ at byte {offset}"));
        }
        let mut trailing = vec![0u8; self.trailer];
        if let Err(e) = self.file.read_exact(&mut trailing) {
            if self.allow_orphaned_tail && e.kind() == std::io::ErrorKind::UnexpectedEof {
                let got = self.file.stream_position()? - (offset + 1024 + header.n as u64 * 8);
                self.truncated_trailer_missing_bytes += self.trailer as u64 - got;
                eprintln!(
                    "WARNING: incomplete unused trailer at byte {offset}; declared IQ retained"
                );
            } else {
                return Err(e).with_context(|| format!("truncated trailer at byte {offset}"));
            }
        }
        let words: Vec<i32> = payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        // Recorder stores (I,Q); both supplied readers define voltage as Q + i I.
        let iq = words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| Complex32::new(p[1] as f32, p[0] as f32))
            .collect();
        Ok(Some(Frame {
            offset,
            header,
            header_bytes: b,
            iq,
            words,
        }))
    }
    fn finish_tail(&mut self, offset: u64) -> Result<Option<Frame>> {
        let end = self.file.seek(SeekFrom::End(0))?;
        self.orphaned_tail = Some((offset, end - offset));
        eprintln!("WARNING: {} bytes of unheadered/incomplete EOF data at byte {offset}; source retained, no timestamp assigned",end-offset);
        Ok(None)
    }
}

#[cfg(test)]
pub fn fixture_header(n: usize, frac: u16) -> [u8; 1024] {
    let mut h = [0u8; 1024];
    h[..4].copy_from_slice(&0xAAAA5555u32.to_le_bytes());
    h[1020..].copy_from_slice(&0xAA5555AAu32.to_le_bytes());
    h[12..18].copy_from_slice(&[11, 26, 19, 17, 0, 46]);
    h[18..20].copy_from_slice(&frac.to_le_bytes());
    h[36..38].copy_from_slice(&0x1000u16.to_le_bytes());
    h[764] = 1;
    h[766..768].copy_from_slice(&466u16.to_le_bytes());
    h[768..772].copy_from_slice(&(n as u32).to_le_bytes());
    h[980..982].copy_from_slice(&5000u16.to_le_bytes());
    h[988..990].copy_from_slice(&200u16.to_le_bytes());
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn clock_and_rejections() {
        let mut b = fixture_header(1000, 200);
        let h = Header::parse(&b, 8).unwrap();
        assert_eq!(h.utc_ns, 1794915960005000000);
        b[764] = 2;
        assert!(Header::parse(&b, 8).is_err());
        b[764] = 1;
        b[12] = 13;
        assert!(Header::parse(&b, 8).is_err());
    }
    #[test]
    fn payload_order_false_magic_and_truncation() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for k in 0..2 {
            f.write_all(&fixture_header(800, k * 200)).unwrap();
            for _ in 0..800 {
                f.write_all(&0xAAAA5555u32.to_le_bytes()).unwrap();
                f.write_all(&375i32.to_le_bytes()).unwrap();
            }
            f.write_all(&[0u8; 8]).unwrap();
        }
        let mut r = Reader::open(f.path(), 8, 8).unwrap();
        assert_eq!(
            r.next().unwrap().unwrap().iq[0],
            Complex32::new(375.0, 0xAAAA5555u32 as i32 as f32)
        );
        assert_eq!(r.next().unwrap().unwrap().offset, 7432);
        assert!(r.next().unwrap().is_none());
        f.write_all(&[0u8; 3]).unwrap();
        assert!(r.next().is_err());
    }
    #[test]
    fn bounded_recovery_records_skipped_bytes() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 4096]).unwrap();
        f.write_all(&fixture_header(800, 200)).unwrap();
        f.write_all(&vec![0u8; 6408]).unwrap();
        assert!(Reader::open(f.path(), 8, 8).unwrap().next().is_err());
        let mut r = Reader::open(f.path(), 8, 8).unwrap();
        r.max_resync_bytes = 4096;
        assert_eq!(r.next().unwrap().unwrap().offset, 4096);
        assert_eq!(r.recoveries, vec![(0, 4096)]);
        assert!(r.next().unwrap().is_none());
        let mut r = Reader::open(f.path(), 8, 8).unwrap();
        r.max_resync_bytes = 4092;
        assert!(r.next().is_err());
    }
    #[test]
    fn orphan_tail_policy_is_explicit() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&vec![0; 6408]).unwrap();
        let mut r = Reader::open(f.path(), 8, 8).unwrap();
        r.max_resync_bytes = 10000;
        assert!(r.next().is_err());
        let mut r = Reader::open(f.path(), 8, 8).unwrap();
        r.max_resync_bytes = 10000;
        r.allow_orphaned_tail = true;
        assert!(r.next().unwrap().is_none());
        assert_eq!(r.orphaned_tail, Some((0, 6408)));
    }
    #[test]
    fn retains_full_iq_when_only_unused_trailer_is_truncated() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&fixture_header(800, 0)).unwrap();
        f.write_all(&vec![0; 6400]).unwrap();
        let mut r = Reader::open(f.path(), 8, 8).unwrap();
        r.allow_orphaned_tail = true;
        assert_eq!(r.next().unwrap().unwrap().iq.len(), 800);
        assert_eq!(r.truncated_trailer_missing_bytes, 8);
        assert!(r.next().unwrap().is_none());
    }
}
