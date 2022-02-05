using ArgParse
using Random
using Strudel

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "dataset"
            help = "Dataset name"
            required = true
        "--pseudocount"
            arg_type = Float64
            default = 1.0
        "--maxiter", "-n"
            help = "Number of iterations"
            arg_type = Int64
            default = 500
        "--seed"
            help = "Seed for random generation"
            arg_type = Int
            default = 1337
        "--pc_path"
            help = "Single compoents to initialize EM mixtures"
            arg_type = String
            required = true
        "--num_mix"
            help = "Number of mixtures in EM"
            arg_type = Int64
            default = 2
    end
    return parse_args(ARGS, s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_cmd()
    Random.seed!(args["seed"])
    main_learn_mixture(args["dataset"], args["num_mix"];
                       maxiter=args["maxiter"], 
                       pseudocount=args["pseudocount"],
                       pc_file=args["pc_path"])
end


