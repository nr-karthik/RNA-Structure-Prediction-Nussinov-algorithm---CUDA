import os

# Paths
INPUT_FASTA = "data/bpRNA_1m_90.fasta"
OUTPUT_DIR  = "data/baseline_bins"

# Upper bounds for each bin
BINS = [256, 512, 1024, 4096, 8192]

def get_bin_name(length):
    for b in BINS:
        if length <= b:
            return f"seq_{b}.fa"
    return "seq_large.fa"

def save_to_bin(header, sequence):
    fname = get_bin_name(len(sequence))
    path  = os.path.join(OUTPUT_DIR, fname)
    with open(path, 'a') as out_f:
        out_f.write(f"{header}\n{sequence}\n")

def bin_sequences():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear stale bin files before writing
    for b in BINS:
        p = os.path.join(OUTPUT_DIR, f"seq_{b}.fa")
        if os.path.exists(p):
            os.remove(p)
    large_p = os.path.join(OUTPUT_DIR, "seq_large.fa")
    if os.path.exists(large_p):
        os.remove(large_p)

    print(f"Reading {INPUT_FASTA} and sorting into bins...")

    current_header = ""
    current_seq    = []   # <-- was missing []
    count = 0

    with open(INPUT_FASTA, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save the previous sequence before starting a new one
                if current_header and current_seq:
                    save_to_bin(current_header, "".join(current_seq))
                    count += 1
                current_header = line
                current_seq    = []   # <-- was missing []
            else:
                current_seq.append(line)

        # Save the last sequence in the file
        if current_header and current_seq:
            save_to_bin(current_header, "".join(current_seq))
            count += 1

    # Print a summary of what landed in each bin
    print(f"\nDone! {count} sequences sorted into {OUTPUT_DIR}/")
    print("\nBin summary:")
    for b in BINS:
        p = os.path.join(OUTPUT_DIR, f"seq_{b}.fa")
        if os.path.exists(p):
            with open(p) as f:
                lines = f.readlines()
            n_seqs = sum(1 for l in lines if l.startswith(">"))
            print(f"  1–{b:>5} nt  →  seq_{b}.fa  ({n_seqs} sequences)")
        else:
            print(f"  1–{b:>5} nt  →  seq_{b}.fa  (0 sequences)")
    large_p = os.path.join(OUTPUT_DIR, "seq_large.fa")
    if os.path.exists(large_p):
        with open(large_p) as f:
            lines = f.readlines()
        n_seqs = sum(1 for l in lines if l.startswith(">"))
        print(f"  8193+     nt  →  seq_large.fa  ({n_seqs} sequences)")
    else:
        print(f"  8193+     nt  →  seq_large.fa  (0 sequences)")

if __name__ == "__main__":
    bin_sequences()