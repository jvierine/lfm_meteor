use crate::Pulse;

pub fn valid_sequence(utc: &[i64]) -> Vec<u8> {
    let mut tails: Vec<usize> = Vec::new();
    let mut prev = vec![usize::MAX; utc.len()];
    for (i, t) in utc.iter().enumerate() {
        let k = tails.partition_point(|&j| utc[j] < *t);
        if k > 0 {
            prev[i] = tails[k - 1];
        }
        if k == tails.len() {
            tails.push(i);
        } else {
            tails[k] = i;
        }
    }
    let mut flags = vec![0; utc.len()];
    if let Some(&last) = tails.last() {
        let mut i = last;
        loop {
            flags[i] = 1;
            if prev[i] == usize::MAX {
                break;
            }
            i = prev[i];
        }
    }
    flags
}
pub fn mark(pulses: &mut [Pulse]) {
    let flags = valid_sequence(&pulses.iter().map(|p| p.utc).collect::<Vec<_>>());
    for (p, flag) in pulses.iter_mut().zip(flags) {
        p.timing_valid = flag;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quarantines_future_first_header_without_inventing_time() {
        let t = vec![
            100, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
            105,
        ];
        let flags = valid_sequence(&t);
        assert_eq!(flags[0], 0);
        assert!(flags[1..].iter().all(|v| *v == 1));
        assert_eq!(valid_sequence(&[0, 5, 10, 25, 30]), vec![1; 5]);
    }
}
