import argparse

def parse_output(filename):
    results = []
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        name      = lines[i];      i += 1
        seq       = lines[i];      i += 1
        structure = lines[i];      i += 1
        score     = int(lines[i]); i += 1
        results.append((name, seq, structure, score))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, help='GPU output file')
    parser.add_argument('--ref', required=True, help='CPU reference file')
    args = parser.parse_args()

    gpu_results = parse_output(args.gpu)
    ref_results = parse_output(args.ref)

    if len(gpu_results) != len(ref_results):
        print(f"ERROR: sequence count mismatch — GPU:{len(gpu_results)} REF:{len(ref_results)}")
        return

    passed = 0
    failed = 0

    for i, (gpu, ref) in enumerate(zip(gpu_results, ref_results)):
        gpu_name, gpu_seq, gpu_struct, gpu_score = gpu
        ref_name, ref_seq, ref_struct, ref_score = ref

        # normalize header — strip > if present
        gpu_name_clean = gpu_name.lstrip('>')
        ref_name_clean = ref_name.lstrip('>')

        if gpu_name_clean != ref_name_clean:
            print(f"WARNING: name mismatch at index {i}: GPU='{gpu_name_clean}' REF='{ref_name_clean}'")

        if gpu_score == ref_score:
            print(f"[PASS] {ref_name_clean:<30} score={gpu_score}")
            passed += 1
        else:
            print(f"[FAIL] {ref_name_clean:<30} GPU={gpu_score} REF={ref_score}")
            print(f"       seq: {gpu_seq[:50]}...")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Result: {passed}/{passed+failed} sequences passed")
    if failed == 0:
        print("✅ All scores match — GPU kernel is correct")
    else:
        print("❌ Score mismatches detected — check kernel logic")

if __name__ == '__main__':
    main()
