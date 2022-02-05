export main_greedy_search, main_beam_search

################################################################################
# Greedy Search 
################################################################################

function main_greedy_search(name::String;
                            pick_edge="eFlow", 
                            pick_var="vMI", 
                            depth=1,
                            pseudocount=1.0,
                            maxiter=5000, 
                            max_circuit_size=150_000, 
                            max_learning_time=24*60*60,
                            patience=100,
                            outdir=tempdir())
    
    logger = ConsoleLogger(stdout)
    global_logger(logger)

    @info "Loading dataset $name"
    train_x, valid_x, test_x = twenty_datasets(name)

    @info "Greedy search with heuristics for $maxiter iterations"
    log = greedy_log(train_x, valid_x, test_x; 
        outdir, patience, pseudocount, max_circuit_size, max_learning_time)

    greedy_search(train_x; pick_edge, pick_var, depth, pseudocount, 
        log_per_iter=log, maxiter)
end


function greedy_search(train_x; 
                        pick_edge="eFlow", 
                        pick_var="vMI", 
                        depth=1, 
                        pseudocount=1.0,
                        maxiter=100,
                        log_per_iter=noop)

    # initialize
    tic = time_ns()
    pc, _ = learn_chow_liu_tree_circuit(train_x)
    estimate_parameters!(pc, train_x; pseudocount)
    toc = time_ns()
    log_per_iter(pc, 0; time=(toc-tic)/1e9)

    # structure_update
    loss(circuit) = heuristic_loss(circuit, train_x; pick_edge, pick_var)
    
    pc_split_step(circuit) = begin
        r = split_step(circuit; loss=loss, depth=depth, sanity_check=false)
        if isnothing(r) return nothing end
        c, = r
        estimate_parameters!(c, train_x; pseudocount)
        return c, missing
    end

    iter = 0

    last_time = time()
    log_iter_caller(circuit) = begin
        stop_flag, _ = log_per_iter(circuit, iter; time=(time() - last_time)/1e9)
        iter += 1
        last_time = time()
        return stop_flag
    end

    pc = struct_learn(pc; 
        primitives=[pc_split_step], kwargs=Dict(pc_split_step=>()), 
        maxiter=maxiter, stop=log_iter_caller, verbose=false)

    pc
end

################################################################################
# Beam Search 
################################################################################

function main_beam_search(name::String, beam_width::Int; 
                          pick_edge="eFlow", 
                          pick_var="vMI", 
                          depth=1,
                          pseudocount=1.0,
                          selection="train", # valid
                          kld_threshold=-1,
                          maxiter=5000, 
                          max_circuit_size=150_000, 
                          max_learning_time=24*60*60,
                          patience=100,
                          outdir=tempdir())
    
    logger = ConsoleLogger(stdout)
    global_logger(logger)

    @info "Loading dataset $name"
    train_x, valid_x, test_x = twenty_datasets(name)

    @info "Beam search with beam_width = $beam_width for $maxiter iterations"
    log = beam_log(train_x, valid_x, test_x; 
        patience, outdir, pseudocount, max_circuit_size, max_learning_time)

    beam_search(train_x, beam_width; pick_edge, pick_var, depth, pseudocount,
            log_per_iter=log, maxiter, selection, valid_x, kld_threshold,
            max_circuit_size)
end


function beam_search(train_x, beam_width;
                     log_per_iter=noop,
                     pick_edge="eFlow", 
                     pick_var="vMI", 
                     depth=1, 
                     pseudocount=1.0,
                     maxiter=100,
                     selection="train",
                     valid_x=nothing,
                     kld_threshold=-1.0,
                     max_circuit_size=150_000)

    # initialize
    tic = time_ns()
    pc, _ = learn_chow_liu_tree_circuit(train_x)
    estimate_parameters!(pc, train_x; pseudocount)
    toc = time_ns()
    log_per_iter(pc, 0, 1; time=(toc-tic)/1e9, max_beam_idx=1)

    # beam search
    beam_set = [pc]
    stop_flag = false
    for iter in 1 : maxiter
        @info "\nIteration $iter/$maxiter"
        tic = time_ns()
        beam_set = update_beam_set(train_x, beam_set, beam_width;
            pick_edge, pick_var, depth, pseudocount, selection,
            valid_x, kld_threshold, max_circuit_size)
        toc = time_ns()
        
        for (i, pc_i) in enumerate(beam_set)
            @info "Beam circuit $i/$(length(beam_set)): "
            stop_flag, pc = log_per_iter(pc_i, iter, i; time=(toc-tic)/1e9, max_beam_idx=length(beam_set))
        end

        if stop_flag || isempty(beam_set)
            @info "Training is finished on iteration $iter according to early stopping"
            return pc
        end
    end
    return pc
end


using ProbabilisticCircuits: vMI
function update_beam_set(train_x, beam_set, beam_width; 
                         pick_edge="eFlow", 
                         pick_var="vMI", 
                         depth=1, 
                         pseudocount=1.0,
                         selection="train", 
                         valid_x=nothing, 
                         kld_threshold=-1.0, 
                         max_circuit_size=150_000)

    edges_all = []
    flows_all = []
    circuit_idx = []
    variable_scope_all = Dict()

    @assert pick_edge === "eFlow" && pick_var == "vMI" "($pick_edge, $pick_var) is not supported by beam search."

    for (idx, circuit) in enumerate(beam_set)
        candidates, variable_scope = split_candidates(circuit)
        values, flows, node2id = satisfies_flows(circuit, train_x)

        # pick_edge = eFlow
        edges, flows = eFlow(values, flows, candidates, node2id; num_best=beam_width)
        append!(edges_all, edges)
        append!(flows_all, flows)
        append!(circuit_idx, idx * ones(Int, beam_width))
        merge!(variable_scope_all, variable_scope)
    end

    idx = partialsortperm(flows_all, 1 : min(beam_width^2, length(flows_all)), rev=true)

    new_circuits = []
    for i in idx
        edge = edges_all[i]
        flow = flows_all[i]
        circuit = beam_set[circuit_idx[i]]

        # pick var = vMI
        or, and = edge
        vars = Var.(collect(variable_scope_all[and]))
        values, flows, node2id = satisfies_flows(circuit, train_x; weights=nothing) # TODO optimize this duplicated line
        var, score = vMI(values, flows, edge, vars, train_x, node2id)

        new_circuit, _ = split(circuit, edge, var; depth)
        if num_nodes(new_circuit) < max_circuit_size
            push!(new_circuits, new_circuit)
        end
    end

    # remove duplicated circuits in the candidates
    distinct_new_circuits = []
    params = Set()
    lls = []

    for circuit in new_circuits
        data = (selection == "valid" ? valid_x : train_x) 
        estimate_parameters!(circuit, train_x; pseudocount)
        para = (num_nodes(circuit), num_edges(circuit), num_parameters(circuit), log_likelihood_avg(circuit, data))
        if (!(para in params) && !has_similar(circuit, distinct_new_circuits;kld_threshold))
            push!(distinct_new_circuits, circuit)
            push!(params, para)
            push!(lls, para[4])
        end
    end

    # pick the best from B^2 candidates according to log-log_likelihood

    # @info "Circuits $distinct_new_circuits"
    @info "LogLikelihood of all $(length(lls))/$(beam_width^2) candidates" 

    idx = partialsortperm(lls, 1 : min(beam_width, length(lls)), rev=true)

    @info "Pick the ith circuit: $idx"
    
    return distinct_new_circuits[idx]
end


function has_similar(circuit, circuit_set; kld_threshold=-1)
    if kld_threshold < 0.0 || isempty(circuit_set) 
        return false
    else
        for c in circuit_set
            if kl_divergence(c, circuit) < kld_threshold
                return true
            end
        end
        return false
    end
end


# copied and modified from ProbabilisticCircuits
# TODO: num_best < beam_width, find beam_width candidates in total, not for each circuit
using LogicCircuits: count_downflow
function eFlow(values, flows, candidates, node2id; num_best=1)
    edge2flows = map(candidates) do (or, and)
        count_downflow(values, flows, nothing, or, and, node2id)
    end
    # if num_best == 1
        # (max_flow, max_edge_id) = findmax(edge2flows)
        # candidates[max_edge_id], max_flow
    # else
        idx = partialsortperm(edge2flows, 1 : min(num_best, length(edge2flows)), rev=true)
        candidates[idx], edge2flows[idx]
    # end
end

