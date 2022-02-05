using Plots
using Logging
using JSON
using IterTools
using Random
using LinearAlgebra
using Printf
using CSV
using Statistics
using DataFrames
using MAT
using Suppressor

export beam_log, greedy_log


function read_dict_from_json(filename)
    return JSON.parse(open(f -> read(f, String), filename))
end


function read_all_exp_configs(dir; given=Dict())
    results = []
    for ite in walkdir(dir)
        (root, dirs, files) = ite
        if !isempty(files) && ("config.json" in files)
            # read config
            config = read_dict_from_json(joinpath(root, "config.json"))
            flag = true
            for (k, v) in given
                if config[k] != v
                    flag = false
                end
            end
            if flag
                config["root"] = root
                push!(results, config)
            end
        end
    end
    return results
end


function select_exp_config(configs, given::Dict)
    results = []
    for config in configs
        flag = true
        for (k, v) in given
            if config[k] != v
                flag = false
            end
        end
        if flag
            push!(results, config)
        end
    end
    return results
end


function read_dict_from_csv(filename; labels=nothing, pick_stop=false, stop_key="stop?")
    dataframe = CSV.read(filename, DataFrame; header=true)
    header = map(x -> String(x), names(dataframe))
    result = Dict()
    if isnothing(labels)
        labels = header
    end
    for label in labels
        @assert label in header

        result[label] = convert(Vector, dataframe[!, Symbol(label)])
    end

    if !pick_stop
        return result
    else
        ite = findall(x -> !ismissing(x) && x, convert(Vector, dataframe[!, Symbol(stop_key)]))
        @info ite
        if issomething(ite)
            return map(labels) do label
                result[label][ite]
            end
        else
            @info "The value for $stop_key == true is not found!"
            return result
        end
    end
    
end


function save_as_csv(dict::Dict; filename, header=keys(dict))
    dir = dirname(filename)
    if !isdir(dir)
        mkpath(dir)
    end
    table = DataFrame(;[Symbol(x) => dict[x] for x in header]...)
    CSV.write(filename, table; )
    table
end


function extend_dict_to_same_length(dict::Dict)
    len = 0
    for (k, v) in dict
        len = maximum([len, length(v)])
    end
    for (k, v) in dict
        dict[k] = [dict[k]; missings(len - length(v))]
    end
    len
end


function arg2str(args)
    out = @capture_out foreach(x -> println(x[1], " => ", x[2]), args)
end


"""
Initialize dictionary with empty contents
"""
function dict_init(header;length=0)
    results = Dict()
    for x in header
        results[x] = Vector{Union{Any,Missing}}(missing, length)
    end
    results
end


"""
Initialize log dict
"""
function log_init(header)
    dict_init(header)
end


"""
Every log step
"""
function dict_append!(dict, header, value)
    for(k, v) in zip(header, value)
        push!(dict[k], v)
    end
end


function greedy_log(train_x, valid_x, test_x;
    outdir, patience, pseudocount, max_circuit_size, max_learning_time)
    header_all = ["iter", "time", "total_time", "#nodes", "#edges", "#paras", 
                  "train_ll", "valid_ll", "test_ll", "stop?"]

    dict_all = log_init(header_all)

    if !ispath(outdir)
        mkpath(outdir)
    end

    best_circuit = nothing

    function logger(circuit, iter; time=0)

        # save circuit log
        nn = num_nodes(circuit)
        ne = num_edges(circuit)
        np = num_parameters(circuit)

        # need to do estimate_parameters! for each circuit in the beam set
        # since they do not necessary share the same parameters
        ll1 = log_likelihood_avg(circuit, train_x)
        ll2 = log_likelihood_avg(circuit, valid_x)
        ll3 = log_likelihood_avg(circuit, test_x)
        if iter == 0
            tt = time
            best_circuit = circuit
        else
            tt = dict_all["total_time"][end] + time
        end

        dict_append!(dict_all, header_all, [iter, time, tt, nn, ne, np, ll1, ll2, ll3, missing])
        @info "Iteration $iter. train_LL = $(ll1); valid_LL  = $(ll2); test_LL = $(ll3); #nodes = $nn; #edges = $ne; #params = $np"

        save_as_csv(dict_all; filename=joinpath(outdir, "progress_all.csv"), header=header_all)

        # save best_circuit, dict_best, return true/false
        stop_flag, best_circuit = isstop(iter, circuit, dict_all, best_circuit, tt; 
            patience, max_circuit_size, max_learning_time)
        estimate_parameters!(best_circuit, train_x; pseudocount) # TODO remove
        write((joinpath(outdir, "best.psdd"), joinpath(outdir, "best.vtree")), best_circuit)

        return stop_flag, best_circuit    
    end
end


function beam_log(train_x, valid_x, test_x; 
        outdir, patience, pseudocount, max_circuit_size, max_learning_time)
    header_all = ["iter", "beam_idx", "#nodes", "#edges", "#paras", "train_ll", "valid_ll", "test_ll"]
    dict_all = log_init(header_all)
    header_best = ["iter", "time", "total_time", "#nodes", "#edges", "#paras", "train_ll", "valid_ll", "test_ll", "stop?"]
    dict_best = log_init(header_best)
    if !ispath(outdir)
        mkpath(outdir)
    end

    log_tmp = []
    # best_circuit_path = (joinpath(outdir, "best.psdd"), joinpath(outdir, "best.vtree"))
    best_circuit = nothing

    function logger(circuit, iter, beam_idx; time=0, max_beam_idx)
        if beam_idx == 1
            log_tmp = []
        end

        # @info "Length of log_tmp $(length(log_tmp))"
        # save all beam circuit log
        nn = num_nodes(circuit)
        ne = num_edges(circuit)
        np = num_parameters(circuit)

        # need to do estimate_parameters! for each circuit in the beam set
        # since they do not necessary share the same parameters
        estimate_parameters!(circuit, train_x; pseudocount) 
        ll1 = log_likelihood_avg(circuit, train_x)
        ll2 = log_likelihood_avg(circuit, valid_x)
        ll3 = log_likelihood_avg(circuit, test_x)

        dict_append!(dict_all, header_all, [iter, beam_idx, nn, ne, np, ll1, ll2, ll3])
        @info "Iteration $iter. train_LL = $(ll1); valid_LL  = $(ll2); test_LL = $(ll3); #nodes = $nn; #edges = $ne; #params = $np"
        push!(log_tmp, (nn, ne, np, ll1, ll2, ll3, circuit))

        save_as_csv(dict_all; filename=joinpath(outdir, "progress_all.csv"), header=header_all)

        # save best_circuit, dict_best, return true/false

        stop_flag = false
        if iter == 0
            stop = missing
            tt = time
            @assert length(log_tmp) == 1
            nn, ne, np, ll1, ll2, ll3, circuit = log_tmp[1]
            dict_append!(dict_best, header_best, [iter, time, tt, nn, ne, np, ll1, ll2, ll3, stop])
            log_tmp = []
            best_circuit = circuit

        elseif beam_idx == max_beam_idx
            stop = missing
            tt = dict_best["total_time"][end] + time
            @assert length(log_tmp) == max_beam_idx
            ll, id = findmax(map(x -> x[4], log_tmp))
            nn, ne, np, ll1, ll2, ll3, circuit_iter = log_tmp[id]
            dict_append!(dict_best, header_best, [iter, time, tt, nn, ne, np, ll1, ll2, ll3, stop])
            log_tmp = []
            stop_flag, best_circuit = isstop(iter, circuit_iter, dict_best, best_circuit, tt; patience, max_circuit_size, max_learning_time)

        else
            stop_flag = false
        end
        save_as_csv(dict_best; filename=joinpath(outdir, "progress_best.csv"), header=header_best)

        estimate_parameters!(best_circuit, train_x; pseudocount) # TODO remove
        write((joinpath(outdir, "best.psdd"), joinpath(outdir, "best.vtree")), best_circuit)

        return stop_flag, best_circuit
    
    end
end


function isstop(cur_iter, cur_circuit, dict_best, best_circuit, total_time_taken; 
    patience, max_circuit_size=nothing, max_learning_time=nothing)
    max_ll, max_id = findmax(dict_best["valid_ll"])
    max_iter = dict_best["iter"][max_id]

    # pre_best_circuit = load_struct_prob_circuit(best_circuit_path[1], best_circuit_path[2])[1]
    # pre_vtree = pre_best_circuit.vtree

    # best circuit, stop?  stop_flag
    if max_iter == cur_iter
        new_best_circuit = cur_circuit
        vtree = cur_circuit.vtree
        dict_best["stop?"] .= missing
        dict_best["stop?"][end] = true
    else
        new_best_circuit = best_circuit
        vtree = best_circuit.vtree
    end

    stop_flag = false
    if cur_iter - max_iter >= patience
        stop_flag = true
    end

    if !isnothing(max_circuit_size) && num_nodes(cur_circuit) > max_circuit_size
        stop_flag = true; # stop
    end

    if !isnothing(max_learning_time) && total_time_taken > max_learning_time
        stop_flag = true; # stop
    end


    # @info "Best circuit ll in is is $(log_likelihood_avg(new_best_circuit, valid_x; use_gpu=false))"
    return stop_flag, new_best_circuit
end