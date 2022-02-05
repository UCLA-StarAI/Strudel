export main_learn_mixture


function main_learn_mixture(name::String, num_mix;
                       maxiter=500, 
                       pseudocount=1.0,
                       pc_file)

    logger = ConsoleLogger(stdout)
    global_logger(logger)

    @info "Loading dataset $name"
    train_x, valid_x, test_x = twenty_datasets(name)
    pc = read(pc_file, ProbCircuit)
    train_mixture_circuit(pc, train_x; num_mix, pseudocount,
        em_maxiter = maxiter, valid_x, test_x)
end


using ProbabilisticCircuits: one_step_em, update_pc_params_from_pbc!
function train_mixture_circuit(pc, data;
                                num_mix=5,
                                pseudocount=1.0,
                                em_maxiter=500,
                                valid_x=nothing,
                                test_x=nothing)

    spc = compile(SharedProbCircuit, pc, num_mix)
    values, flows, node2id = satisfies_flows(spc, data)
    component_weights = reshape(initial_weights(data, num_mix), 1, num_mix)
    update_pc_params_from_pbc!(spc, ones(Float64, num_examples(data), num_mix) ./ num_mix, 
                               values, flows, node2id; pseudocount)

    ll1 = nothing
    for iter in 1 : em_maxiter
        @assert isapprox(sum(component_weights), 1.0; atol=1e-10)
        ll1, component_weights = one_step_em(spc, data, values, flows, node2id, 
                component_weights; pseudocount)
        ll2 = log_likelihood_avg(spc, valid_x)
        ll3 = log_likelihood_avg(spc, test_x)
        @info "Iteration $iter/$em_maxiter. train_LL = $(mean(ll1)); valid_LL  = $(ll2); test_LL = $(ll3)"
    end
    spc, component_weights, ll1
end