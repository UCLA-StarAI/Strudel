using ArgParse
using Random
using Strudel

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "dataset"
            help = "Dataset name"
            required = true
        "--search"
            help = "Search stratrgy"
            default = "greedy"
            range_tester = x -> (x in (["greedy", "beam"]))
        "--pick_edge"
            arg_type = String
            default = "eFlow"
            range_tester = x -> (x in (["eFlow", "eRand"]))
        "--pick_var"
            arg_type = String
            default = "vMI"
            range_tester = x -> (x in (["vMI", "vRand"]))
        "--depth"
            arg_type = Int64
            default = 1
        "--pseudocount"
            arg_type = Float64
            default = 1.0
        "--maxiter", "-n"
            help = "Number of iterations"
            arg_type = Int64
            default = 5000
        "--max_circuit_size"
            help = "Maximum circuit size"
            arg_type = Int64
            default = 150_000
        "--max_learning_time"
            help = "Maximum learning time"
            arg_type = Int64
            default = 24*60*60
        "--patience"
            arg_type = Int64
            default = 100
        "--outdir", "-d"
            help = "Output directory"
            arg_type = String
            default = "exp/"
        "--seed"
            help = "Seed for random generation"
            arg_type = Int
            default = 1337
        # beam seach 
        "--beam_width"
            help = "Wdith for beam search"
            arg_type = Int
            default = 5
        "--selection"
            help = "Use train/valid set for selectiong best model from candidates"
            arg_type = String
            default = "train"
            range_tester = x -> (x in (["train", "valid"]))
        "--kld_threshold"
            help = "Comparing whether two circuits are similar via KL divergence"
            arg_type = Float64
            default = -1.0
    end
    return parse_args(ARGS, s)
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_cmd()
    Random.seed!(args["seed"])
    if args["search"] == "greedy"
        main_greedy_search(args["dataset"];
                pick_edge=args["pick_edge"], 
                pick_var=args["pick_var"], 
                depth=args["depth"],
                pseudocount=args["pseudocount"],
                maxiter=args["maxiter"], 
                max_circuit_size=args["max_circuit_size"], 
                max_learning_time=args["max_learning_time"],
                patience=args["patience"],
                outdir=args["outdir"])
    elseif args["search"] == "beam"
        main_beam_search(args["dataset"], args["beam_width"];
            pick_edge=args["pick_edge"], 
            pick_var=args["pick_var"], 
            depth=args["depth"],
            pseudocount=args["pseudocount"],
            maxiter=args["maxiter"], 
            max_circuit_size=args["max_circuit_size"], 
            max_learning_time=args["max_learning_time"],
            patience=args["patience"],
            outdir=args["outdir"],
            
            selection=args["selection"],
            kld_threshold=args["kld_threshold"])
    else
        @assert false "$(args["search"]) not found."
    end
end



