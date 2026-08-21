import OMEinsumContractionOrders, Random, Printf

function flushprintln(msg)
    println(msg)
    flush(stdout)
end

function unique_preserve_order(arr)
    seen = Set{eltype(arr)}()
    unique_arr = eltype(arr)[]
    for item in arr
        if !(item in seen)
            push!(seen, item)
            push!(unique_arr, item)
        end
    end
    return unique_arr
end

function qec_from_eqns(filename)
    data = open(filename, "r") do f
        read(f, String)
    end

    clean_data = strip(data)
    eqns = split(clean_data, "->")[1]
    inputs_data = split(eqns, ",")

    unique_chars = Set(Iterators.flatten(inputs_data))
    sorted_unique_chars = sort(collect(unique_chars))
    char_to_int_map = Dict(c => i for (i, c) in enumerate(sorted_unique_chars))
    mapped_inputs = [unique_preserve_order([char_to_int_map[c] for c in sublist]) for sublist in inputs_data]

    rhs = split(clean_data, "->")[2]
    output_char = rhs[1]
    output = [char_to_int_map[output_char]]

    return OMEinsumContractionOrders.EinCode(mapped_inputs, output), output[1]
end

# Get args
eq_file = ARGS[1]
out_file = ARGS[2]
seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
ntrials = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 192
niters = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20
sc_target = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 33.0
sc_weight = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.5
rw_weight = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 1024.0
beta_start = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0
beta_stop = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 10.0
use_slicer = length(ARGS) >= 11 ? lowercase(ARGS[11]) != "noslicer" : true
init_tree_file = length(ARGS) >= 12 ? ARGS[12] : ""
beta_steps = length(ARGS) >= 13 ? parse(Int, ARGS[13]) : 2
tc_weight = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 1.0
initializer_name = length(ARGS) >= 15 ? lowercase(ARGS[15]) : ""
decomposition_name = length(ARGS) >= 16 ? lowercase(ARGS[16]) : "tree"
betas = beta_steps <= 2 ? [beta_start, beta_stop] : collect(range(beta_start, beta_stop, length=beta_steps))

flushprintln("Threads: $(Threads.nthreads())")
flushprintln("Reading equation: $eq_file")

code, output = qec_from_eqns(eq_file)
flushprintln("Code ixs length: $(length(code.ixs)), iy: $(code.iy)")
size_dict = OMEinsumContractionOrders.uniformsize(code, 2)
flushprintln("Size dict: $(length(size_dict)) entries")

Random.seed!(seed)
score = OMEinsumContractionOrders.ScoreFunction(
    tc_weight=tc_weight,
    sc_target=sc_target,
    sc_weight=sc_weight,
    rw_weight=rw_weight,
)
optimizer_input = code
initializer = :greedy
decomposition_type = decomposition_name == "path" ? OMEinsumContractionOrders.PathDecomp() : OMEinsumContractionOrders.TreeDecomp()
if initializer_name == "random"
    initializer = :random
elseif initializer_name == "greedy"
    initializer = :greedy
end
if init_tree_file != ""
    loaded_tree = OMEinsumContractionOrders.readjson(init_tree_file)
    optimizer_input = loaded_tree
    if hasproperty(loaded_tree, :slicing)
        optimizer_input = getproperty(loaded_tree, :eins)
    end
    initializer = :specified
    flushprintln("Initial tree: $init_tree_file")
end

flushprintln("Starting TreeSA optimization (seed=$seed, ntrials=$ntrials, niters=$niters, threads=$(Threads.nthreads()))...")
flushprintln("score: tc_weight=$(score.tc_weight), sc_target=$(score.sc_target), sc_weight=$(score.sc_weight), rw_weight=$(score.rw_weight)")
flushprintln("betas: [$beta_start, $beta_stop], steps=$(length(betas))")
flushprintln("slicer: $(use_slicer ? "TreeSASlicer" : "disabled")")
flushprintln("initializer: $initializer")
flushprintln("decomposition: $(decomposition_name)")

t_exec = @elapsed begin
    if use_slicer
        optcode_tree = OMEinsumContractionOrders.optimize_code(optimizer_input, size_dict,
            OMEinsumContractionOrders.TreeSA(; βs=betas, ntrials=ntrials, niters=niters, score, initializer, decomposition_type);
            slicer=OMEinsumContractionOrders.TreeSASlicer(; score)
        )
    else
        optcode_tree = OMEinsumContractionOrders.optimize_code(optimizer_input, size_dict,
            OMEinsumContractionOrders.TreeSA(; βs=betas, ntrials=ntrials, niters=niters, score, initializer, decomposition_type)
        )
    end
end

cc = OMEinsumContractionOrders.contraction_complexity(optcode_tree, size_dict)
num_slices = hasproperty(optcode_tree, :slicing) ? length(getproperty(optcode_tree, :slicing)) : 0
outputs = "cc $(log10(2^cc.tc)), sc $(cc.sc), rwc $(log10(2^cc.rwc)), alpha $(2^(cc.tc-cc.rwc)), num of slices $(num_slices), time $(round(t_exec, digits=2))s"
flushprintln(outputs)

OMEinsumContractionOrders.writejson(out_file, optcode_tree)
flushprintln("JSON saved to $out_file")
