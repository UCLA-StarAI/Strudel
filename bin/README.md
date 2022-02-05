# Experiments

## Dataset
The twenty datasets benchmark for density estimation is from [here](https://github.com/UCLA-StarAI/Density-Estimation-Datasets). 
## Usage
- Run `bin/single.jl` or `bin/mixture.jl` with `--help` argument to see the usage message. Here are some sample scripts.
  - To learn a single circuit via greedy search or beam serch
  ```
  julia --project=. bin/single.jl nltcs --search greedy
  julia --project=. bin/single.jl nltcs --search beam --beam_width 4
  ```
  - To learn a mixture of circuits
  ```
  julia bin/mixture.jl nltcs --pc_path exp/nltcs.psdd --num_mix 5
  ```
- To generate batches scripts, run

```
julia --project=. bin/gen_exp.jl scripts/gready.json -b bin/single.jl
julia --project=. bin/gen_exp.jl scripts/beam.json -b bin/single.jl
```

