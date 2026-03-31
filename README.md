# Install dependencies
pip install numpy scipy

# Run small illustrative examples
python main.py --examples

# Run on 10k dataset (power iteration, sensitivity analysis, top-10)
python main.py --dataset web-Google_10k.txt --p 0.15 --top 10

# Also compute closed-form (only feasible for smaller graphs)
python main.py --dataset web-Google_10k.txt --closed-form --no-sensitivity

# Run on full web-Google dataset (875k nodes, ~12s total)
python main.py --dataset web-Google.txt.gz --p 0.15 --top 10

# Run AI crawler demo
python main.py --crawler-demo

# Generate convergence and sensitivity plots (requires matplotlib)
python main.py --dataset web-Google_10k.txt --plots
