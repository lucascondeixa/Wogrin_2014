
#-------------------------------------------------------------------------------
# Initialization
#-------------------------------------------------------------------------------

# Threads.nthreads()
# addprocs()

###### NOTE: These packages might be needed, uncomment them accordingly
###### to the use:

# Pkg.add("Gurobi")
# Pkg.add("StatPlots")
# Pkg.add("JuMP")
# Pkg.add("GR")
# Pkg.add("Distances")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("StatsBase")

###### NOTE: Sometimes "Compat" package hampers the other packages to be
###### correctly installed
# Pkg.rm("Compat")

###### NOTE: Checking out if the packages are in place
# Pkg.build()
# Pkg.status()
# Pkg.update()

###### NOTE: Standard procedure
# using Clustering
# using GR
# using IndexedTables
# using StatPlots
using Distributed
@everywhere using SharedArrays
@everywhere using StatsBase
@everywhere using Statistics
@everywhere using Gurobi
@everywhere using JuMP
@everywhere using DataFrames
@everywhere using Distances
@everywhere using CSV

function wogrin2D(k::Int64,
    weight::Int64,
    a₁::Vector{Float64},
    a₂::Vector{Float64},
    parallel::Bool)
    ## Clustering modelling brought by the article named "A New Approach to
    ## Model Load Levels in Electric Power Systems With High Renewable
    ## Penetration" by Sonja Wogrin et al. (2014)

    # x: Aux Array for concatenating the DataFrame used
    # Data: DataFrame used
    # k: number of classes
    # weight: weight used to account clusters' distances
    #                                                   # Weight = 1: dif_mean
    #                                                   # Weight = 2: dif_min
    #                                                   # Weight = 3: dif_max
    # n: Number of points
    # m: Number of dimensions
    # dist: distances from k-centers (n,k)
    # class: Array of classes (n)
    # weights_ar: Dataframe of costs
    # first: Array of first indexes
    # k_cent: Array of first centroids
    # costs: Array of weighted distances
    # total_cost: Auxiliar var.
    # total_cost_new: Auxiliar var.
    # δ: Auxiliar paramenter

    ## Initial definitions
    n = length(a₁)                              # Number of points
    m = 2                                       # Number of dimensions
    dist = SharedArray(zeros(n,k))              # Array of distances (n,k)
    costs = SharedArray(zeros(n,k))             # Array of costs (n,k)
    first_s = sample(1:n,k)                     # Sampling the first centroids
    k_cent = hcat(a₁,a₂)[first_s,:]             # First centroids
    class = zeros(Int64, n)                     # Array of classes (n)
    weights_ar = [zeros(n) zeros(n) zeros(n)]   # Costs array
    Data = hcat(a₁, a₂)                         # Completing the Data with costs
    c_use = zeros(Bool,k)                       # Array with the using status
    total_cost = 0                              # Starting the auxiliar var.
    total_cost_new = 1                          # Starting the auxiliar var.
    δ = 1e-10                                   # Aux. paramenter
    tol = 1e-12                                 # Tolerance

    # First cost settings (Only for 2D)
    dif_mean = mean(Data[:,1]-Data[:,2])
    dif_min = minimum(Data[:,1]-Data[:,2])
    dif_max = maximum(Data[:,1]-Data[:,2])

    # Defining weights
    for i in 1:n
        weights_ar[i,1] = abs(Data[i,1]-Data[i,2]-dif_mean+δ)
        weights_ar[i,2] = abs(Data[i,1]-Data[i,2]-dif_min+δ)
        weights_ar[i,3] = abs(Data[i,1]-Data[i,2]-dif_max+δ)
    end

    if parallel

        #Assigning classes (parallel)
        while total_cost != total_cost_new

            total_cost = total_cost_new

            # Defining distances
            for i in 1:n
                @sync @distributed for j in 1:k
                    dist[i,j] = evaluate(Euclidean(),
                    Data[i,1:2],k_cent[j,:])
                end
            end

            # Defining costs
            for i in 1:n
                @sync @distributed for j in 1:k
                    costs[i,j] = weights_ar[i,weight]*dist[i,j]
                end
            end

            # Defining classes
            for i in 1:n
                class[i] = findmin(costs[i,:])[2]
            end

            ## Classification used / unused clusters
            for i in 1:k
                if length(class[class[:] .== i]) == 0
                    c_use[i] = false
                else
                    c_use[i] = true
                end
            end

            # Update classes (mean for those used and 0 for the unused)
            for j in 1:k
                if c_use[j]
                    k_cent[j,1] = mean(Data[class[:] .== j,:][:,1])
                    k_cent[j,2] = mean(Data[class[:] .== j,:][:,2])
                else
                    k_cent[j,1] = 0
                    k_cent[j,2] = 0
                end
            end

            total_cost_new = sum(costs[:,:])
        end

    else

        #Assigning classes (non_parallel)
        while total_cost != total_cost_new

            total_cost = total_cost_new

            # Defining distances
            for i in 1:n
                for j in 1:k
                    dist[i,j] = evaluate(Euclidean(),
                    Data[i,1:m],k_cent[j,:])
                end
            end

            # Defining costs
            for i in 1:n
                for j in 1:k
                    costs[i,j] = weights_ar[i,weight]*dist[i,j]
                end
            end

            # Defining classes
            for i in 1:n
                class[i] = findmin(costs[i,:])[2]
            end

            ## Classification used / unused clusters
            for i in 1:k
                if length(class[class[:] .== i]) == 0
                    c_use[i] = false
                else
                    c_use[i] = true
                end
            end

            # Update classes (mean for those used and 0 for the unused)
            for j in 1:k
                if c_use[j]
                    k_cent[j,1] = mean(Data[class[:] .== j,:][:,1])
                    k_cent[j,2] = mean(Data[class[:] .== j,:][:,2])
                else
                    k_cent[j,1] = 0
                    k_cent[j,2] = 0
                end
            end

            total_cost_new = sum(costs[:,:])
        end
    end


    Size_Cluster = zeros(k)

    for i in 1:k
    	Size_Cluster[i] = length(Data[class[:] .== i,:][1])
    end

    #-------------------------------------------------------------------------------
    ### Transition Matrix
    #-------------------------------------------------------------------------------

    N_transit = zeros(k,k)
    from = 0
    to = 0
    for i in 1:n-1
    	from = class[i]
    	to = class[i+1]
    	N_transit[from,to] = N_transit[from,to] + 1
    end

    return Data, class, k_cent, N_transit, first_s
end

D = [3300,	2900,	2700,	2500,	2450,	2452,	2500,	2900,	3400,
	3750,	3775,	3750,	3670,	3470,	3350,	3280,	3300,	3600,
    3900,	4050,	3980,	3800,	3850,	3315,	2780,	2480,	2250,
    2100,	2050,	1960,	1910,	2100,	2500,	2825,	2950,	2980,
    2970,	2850,	2790,	2700,	2800,	3100,	3500,	3850,	3750,
    3680,	3450,	2950,	2600,	2400,	2300,	2340,	2680,	3250,
    3710,	4100,	4400,	4510,	4530,	4450,	4150,	4000,	4050,
    3995,	4010,	4300,	4480,	4350,	4100,	3875,	3800,	3375,
    2950,	2700,	2600,	2620,	2900,	3300,	3850,	4100,	4330,
    4400,	4380,	4260,	4000,	3850,	3845,	3900,	3980,	4400,
    4420,	4300,	4000,	3800,	3560,	3200,	2900,	2700,	2620,
    2630,	2875,	3400,	3840,	4145,	4340,	4380,	4350,	4250,
    3930,	3840,	3835,	3900,	3940,	4230,	4320,	4170,	3900,
    3770,	3590,	3155,	2900,	2750,	2640,	2608,	2950,	3390,
    3880,	4040,	4200,	4270,	4238,	4100,	3850,	3710,	3715,
    3740,	3780,	4200,	4240,	4150,	3880,	3730,	3600,	3300,
    2900,	2700,	2625,	2620,	2880,	3220,	3800,	4050,	4230,
    4250,	4230,	4080,	3800,	3700,	3700,	3735,	3800,	4200,
    4250,	4150,	3910,	3700,	3650,	3600.0]

W = [1320,	1310,	1311,	1300,	1290,	1280,	1270,	1260,	1255,
	1260,	1265,	1260,	1270,	1275,	1295,	1310,	1320,	1310,
    1309,	1309,	1325,	1370,	1410,	1440,	1445,	1444,	1415,
    1400,	1390,	1385,	1385,	1360,	1350,	1350,	1350,	1350,
    1350,	1310,	1310,	1310,	1360,	1410,	1460,	1500,	1510,
    1520,	1580,	1630,	1690,	1720,	1730,	1750,	1790,	1850,
    1890,	1885,	1870,	1880,	1890,	1930,	1930,	1930,	1890,
    1830,	1800,	1765,	1745,	1720,	1660,	1630,	1620,	1600,
    1560,	1535,	1500,	1450,	1420,	1430,	1450,	1450,	1435,
    1405,	1385,	1370,	1350,	1320,	1290,	1260,	1240,	1223,
    1200,	1188,	1180,	1173,	1170,	1150,	1120,	1090,	1050,
    1015,	980,	940,	920,	880,	820,	770,	738,	695,
    645,	615,	585,	540,	510,	480,	460,	440,	435,
    415,	400,	405,	400,	380,	360,	358,	358,	359,
    360,	370,	365,	367,	358,	370,	368,	367,	366,
    360,	348,	350,	365,	348,	349,	370,	385,	398,
    398,	395,	377,	358,	345,	330,	330,	330,	315,
    300,	295,	285,	272,	255,	230,	235,	240,	240,
    220,	200,	170,	140,	140,	140.0]

ND = D - W 												# Net demand

tini = time()
wogrin2D(6,3,D,W,false)
tend = time() - tini
println("The time required for the non-parallel processing was: $tend \n")

tini = time()
wogrin2D(6,3,D,W,true)
tend = time() - tini
println("The time required for the parallel processing was: $tend \n")
