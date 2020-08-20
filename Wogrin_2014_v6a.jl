## Optimization modelling brought by the article named
## A New Approach to Model Load Levels in Electric
## Power Systems With High Renewable Penetration
## by Sonja Wogrin et al.

## Indices:
# s,s': System states
# p: Time periods
# g: Generation Unit
#
## Variables:
# q[g,p,s]: Thermal generation decision
# v[p,s]: Wind production, allowing for wind spillage
# u[g,p,s]: Dispatch decisions
# y[g,p,s,s']: Start-up
# z[g,p,s,s']: Shut-down
#
## Parameters:
# T[p,s]: Duration of state s in period p
# N[p,s,s']: Cell in transition matrix that brings the number of state_
# _transitions
# C_up[g]: Start-up cost
# C_dn[g]: Shut-down cost
# CV[g]: Variable cost for unit g
# CF[g]: Fixed cost for unit g
# F[p,s]: Amount of minimum reserves
# D[p,s]: Demand
# V[p,s]: Non-dispatchable renewable power generation level
# Qmax[g]: Upper bound on production
# Qmin[g]: Lower bound on production

#-------------------------------------------------------------------------------
# Initialization
#-------------------------------------------------------------------------------
#Pkg.add("Gurobi")
#using Clustering
#Pkg.add("GR")
#using GR
#using IndexedTables
using Gurobi
using JuMP
using DataFrames
using StatPlots
using Distances
using CSV

#-------------------------------------------------------------------------------
##Input
#-------------------------------------------------------------------------------

Qmin = [650 870 870 870 870 870 900 900 900 900 900 120 120]
Qmax = [1000 1730 1730 1730 1730 1730 2200 2200 2200 2200 2200 500 500]
C_up = [54 30.33 30.33 30.33 30.33 30.33 20.68 20.68 20.68 20.68 20.68 4.96 4.96]
C_dn = C_up/10
CV = [3.99 31.68 31.68 31.68 31.68 31.68 50.08 50.08 50.08 50.08 50.08 72.39 72.39]/1000
CF = [0 0.77 0.77 0.77 0.77 0.77 3.24 3.24 3.24 3.24 3.24 2.07 2.07]

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

#-------------------------------------------------------------------------------
#### Clustering
#-------------------------------------------------------------------------------

function my_cluster(class,weight,a₁,args...)

    if typeof(a₁) != Vector{Float64}
        println("Arg #1 = $a₁ has an inconsistent type")
    end

    #Auxiliar
    x = a₁

    for (i, arg) in enumerate(args)

        #Checking if the sizes of arguments are equal
        if size(a₁) != size(arg)
            j = i+1
            println("Arg #$j = $arg has an inconsistent size")
        end

        #Checking if the types of arguments are consistent
        if typeof(arg) != Vector{Float64}
            j = i+1
            println("Arg #$j = $arg has an inconsistent type")
        end

        # Forming the datafrane structure
        x = [x args[i]]

    end

    ## Initial definitions
    Data = convert(DataFrame,x)
    k = class                                               # Number of clusters
    wgt = weight  		    								# Type of weight used
    #														# Weight = 1: dif_mean
    #														# Weight = 2: dif_min
    #														# Weight = 3: dif_max
    n = nrow(Data)                                          # Number of points
    m = ncol(Data)                                          # Number of dimensions
    dist = zeros(n,k)                                       # Array of distances (n,k)
    class = zeros(Int64, n)                                 # Array of classes (n)
    weights_ar = convert(DataFrame,
    	[zeros(Float64,n) zeros(Float64,n) zeros(Float64,n)])	# Dataframe of costs
    first = sample(1:n,k,replace=false)                     # Sampling the first centroids
    k_cent = sort(Data[first,:])                            # First centroids
    Data = hcat(Data, class)                                # Completing the Data with costs
    costs = zeros(n,k)
    total_cost = 0                                          # Starting the auxiliar var.
    total_cost_new = 1                                      # Starting the auxiliar var.
    δ = 1e-10                                               # Aux. paramenter

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

    #Assigning classes
    while total_cost != total_cost_new

        total_cost = total_cost_new

        # Defining distances
        for i in 1:n
            for j in 1:k
                dist[i,j] = evaluate(Euclidean(),convert(Array,Data[i,1:2]),convert(Array,k_cent[j,:]))
            end
        end

        # Defining costs
        for i in 1:n
            for j in 1:k
                costs[i,j] = weights_ar[i,wgt]*dist[i,j]
            end
        end

        # Defining classes
        for i in 1:n
            Data[i,3] = indmin(costs[i,:])
        end

        # Update classes
        for j in 1:k
            k_cent[j,1] = mean(Data[Data[:,3] .== j,:][:,1])
            k_cent[j,2] = mean(Data[Data[:,3] .== j,:][:,2])
        end

        total_cost_new = sum(costs[:,:])

    end

    Size_Cluster = zeros(k)

    for i in 1:k
    	Size_Cluster[i] = length(Data[Data[:,3] .== i,:][1])
    end

    #-------------------------------------------------------------------------------
    ### Transition Matrix
    #-------------------------------------------------------------------------------

    N_transit = zeros(k,k)
    from = 0
    to = 0
    for i in 1:n-1
    	from = Data[i,3]
    	to = Data[i+1,3]
    	N_transit[from,to] = N_transit[from,to] + 1
    end

    return Data, k_cent, N_transit

end

k = 6
(Data,k_cent,N_transit) = my_cluster(k,3,D,W)
n = nrow(Data)
@df Data scatter(:x1,:x2,color = Data[:,3],label="")
scatter!(k_cent[:,1],k_cent[:,2],c = "Blue",label="")
#Data
#N_transit

Size_Cluster = zeros(k)
for i in 1:k
	Size_Cluster[i] = length(Data[Data[:,3] .== i,:][1])
end

#-------------------------------------------------------------------------------
#### Using Wogrin2014 clusters
#-------------------------------------------------------------------------------

n_Data = hcat(Data,ND,zeros(ND))
for i in 1:n
	n_Data[i,5] = i
end

n_Data = sort(n_Data,cols=(:x1_2))

boundaries = [7 36 33 37 42 13]
b1 = boundaries[1]
b2 = b1 + boundaries[2]
b3 = b2 + boundaries[3]
b4 = b3 + boundaries[4]
b5 = b4 + boundaries[5]
b6 = b5 + boundaries[6]

for i in 1:n
	if i <= b1
		n_Data[i,3] = 1
	end
	if b1 < i <= b2
		n_Data[i,3] = 2
	end
	if b2 < i <= b3
		n_Data[i,3] = 3
	end
	if b3 < i <= b4
		n_Data[i,3] = 4
	end
	if b4 < i <= b5
		n_Data[i,3] = 5
	end
	if b5 < i
		n_Data[i,3] = 6
	end
end

sort!(n_Data,cols=(:x1_3))
#n_Data

#-------------------------------------------------------------------------------
### Cluster Size and k_centers
#-------------------------------------------------------------------------------

n_Size_Cluster = zeros(k)
for i in 1:k
	n_Size_Cluster[i] = length(n_Data[n_Data[:,3] .== i,:][1])
end

n_k_cent = zeros(length(k_cent[:,1]),length(k_cent[1,:]))
for j in 1:k
	n_k_cent[j,1] = mean(n_Data[n_Data[:,3] .== j,:][:,1])
	n_k_cent[j,2] = mean(n_Data[n_Data[:,3] .== j,:][:,2])
end

@df n_Data scatter(:x1,:x2,color = n_Data[:,3],label="")
scatter!(n_k_cent[:,1],n_k_cent[:,2],c = "Blue",label="")

#-------------------------------------------------------------------------------
### Transition Matrix
#-------------------------------------------------------------------------------

N_transit = zeros(k,k)
from = 0
to = 0
for i in 1:n-1
	from = n_Data[i,3]
	to = n_Data[i+1,3]
	N_transit[from,to] = N_transit[from,to] + 1
end

#_______________________________________________________________________________
#-------------------------------------------------------------------------------
#                               Models C,B,A
#-------------------------------------------------------------------------------
#_______________________________________________________________________________


#-------------------------------------------------------------------------------
# Model C formulation
#-------------------------------------------------------------------------------

modelC = Model(solver = GurobiSolver())

#-------------------------------------------------------------------------------
##Sets (Model C)
#-------------------------------------------------------------------------------

G = collect(1:13)           # Thermal Generators
#                           # 1:     Nuclear
#                           # 2-6:   Coal
#                           # 7-11:  Combined Cycle
#                           # 12-13: Fuel oil
P = collect(1:1)            # Number of hours (1 week) = 7*24
S = collect(1:k)            # Number of states
#                           # 1: (High Demand, High Wind)
#                           # 2: (High Demand, Low Wind)
#                           # 3: (Low Demand, High Wind)
#                           # 4: (Low Demand, Low Wind)

#-------------------------------------------------------------------------------
##Parameters (Model C)
#-------------------------------------------------------------------------------
TPS = zeros(P[end],S[end])
VPS = zeros(P[end],S[end])
DPS = zeros(P[end],S[end])
NPSS = zeros(P[end],S[end],S[end])
DPS[1,:] = k_cent[:,1]
VPS[1,:] = k_cent[:,2]
FPS = 0.4*DPS
TPS[1,:] = n_Size_Cluster[:,1]
NPSS[1,:,:] = N_transit

#-------------------------------------------------------------------------------
##Variables (Model C)
#-------------------------------------------------------------------------------

@variable(modelC,q[1:G[end], 1:P[end], 1:S[end]] >= 0)
@variable(modelC,0 <= v[1:P[end], 1:S[end]] <= 2000)
@variable(modelC,u[1:G[end], 1:P[end], 1:S[end]], Bin)
@variable(modelC,0 <= y[1:G[end], 1:P[end], 1:S[end], 1:S[end]] <= 1)
@variable(modelC,0 <= z[1:G[end], 1:P[end], 1:S[end], 1:S[end]] <= 1)

#-------------------------------------------------------------------------------
##Constraints (Model C)
#-------------------------------------------------------------------------------

@constraint(modelC,[g in G, p in P, s in S], q[g,p,s] <= Qmax[g]*u[g,p,s])
@constraint(modelC,[g in G, p in P, s in S], q[g,p,s] >= Qmin[g]*u[g,p,s])
@constraint(modelC,[p in P, s in S], v[p,s] <= VPS[p,s])
@constraint(modelC,[g in G,p in P, s in S, s1 in S], u[g,p,s1] - u[g,p,s] == y[g,p,s,s1] - z[g,p,s,s1])
@constraint(modelC,[p in P, s in S], sum(q[g,p,s] for g in G) == (DPS[p,s] - v[p,s]))
@constraint(modelC,[p in P, s in S], FPS[p,s] <= sum((Qmax[g]*u[g,p,s] - q[g,p,s]) for g in G))

#-------------------------------------------------------------------------------
##Objective Function (Model C)
#-------------------------------------------------------------------------------

@objective( modelC, Min, sum(TPS[p,s]*(CV[g]*q[g,p,s] + CF[g]*u[g,p,s]) for g in G, p in P, s in S) +
                        sum(NPSS[p,s,s1]*(C_up[g]*y[g,p,s,s1] + C_dn[g]*z[g,p,s,s1]) for g in G, p in P, s in S, s1 in S) )
#print(model)
status = solve(modelC)

println("Objective value: ", getobjectivevalue(modelC))
println("q = ", getvalue(q))
println("v = ", getvalue(v))
println("u = ", getvalue(u))
println("y = ", getvalue(y))
println("z = ", getvalue(z))

R_gen_C = zeros(length(n_Data[:,1]),G[end])
for g in G
	for i in 1:length(n_Data[:,1])
		R_gen_C[i,g] = getvalue(q)[g,:,n_Data[i,3]][1]
	end
end

R_dis_C = zeros(length(n_Data[:,1]),G[end])
for g in G
	for i in 1:length(n_Data[:,1])
		R_dis_C[i,g] = getvalue(u)[g,:,n_Data[i,3]][1]
	end
end

R_win_C = zeros(length(n_Data[:,1]))
for i in 1:length(n_Data[:,1])
	R_win_C[i] = getvalue(v)[n_Data[i,3]][1]
end

R_up_C = zeros(length(n_Data[:,1])-1,G[end])
for g in G
	for i in 1:(length(n_Data[:,1])-1)
		R_up_C[i,g] = getvalue(y)[g,:,n_Data[i,3],n_Data[i+1,3]][1]
	end
end

R_down_C = zeros(length(n_Data[:,1])-1,G[end])
for g in G
	for i in 1:(length(n_Data[:,1])-1)
		R_down_C[i,g] = getvalue(z)[g,:,n_Data[i,3],n_Data[i+1,3]][1]
	end
end


#R_gen_C
#R_dis_C
#R_win_C
#R_up_C
#R_down_C

CSV.write("out1_C.csv",convert(DataFrame,[R_gen_C R_dis_C R_win_C]))
CSV.write("out2_C.csv",convert(DataFrame,[R_up_C R_down_C]))



#-------------------------------------------------------------------------------
# Model B formulation
#-------------------------------------------------------------------------------

modelB = Model(solver = GurobiSolver())
nl = k

#-------------------------------------------------------------------------------
##Sets (Model B)
#-------------------------------------------------------------------------------

G = collect(1:13)           # Thermal Generators
#                           # 1:     Nuclear
#                           # 2-6:   Coal
#                           # 7-11:  Combined Cycle
#                           # 12-13: Fuel oil
P = collect(1:1)            # Number of periods
L = collect(1:nl)           # Number of Loads

#-------------------------------------------------------------------------------
##Parameters (Model B)
#-------------------------------------------------------------------------------
TPL = zeros(P[end],L[end])
DPL = zeros(P[end],L[end])
DPL[1,:] = n_k_cent[:,1] - n_k_cent[:,2]
FPL = 0.4*DPL
TPL[1,:] = n_Size_Cluster[:,1]

#-------------------------------------------------------------------------------
##Variables (Model B)
#-------------------------------------------------------------------------------

@variable(modelB,q[1:G[end], 1:P[end], 1:L[end]] >= 0)
@variable(modelB,u[1:G[end], 1:P[end], 1:L[end]], Bin)

#-------------------------------------------------------------------------------
##Constraints (Model B)
#-------------------------------------------------------------------------------

@constraint(modelB,[g in G, p in P, l in L], q[g,p,l] <= Qmax[g]*u[g,p,l])
@constraint(modelB,[g in G, p in P, l in L], q[g,p,l] >= Qmin[g]*u[g,p,l])
@constraint(modelB,[p in P, l in L], sum(q[g,p,l] for g in G) >= DPL[p,l])
@constraint(modelB,[p in P, l in L], FPL[p,l] <= sum((Qmax[g]*u[g,p,l] - q[g,p,l]) for g in G))

#-------------------------------------------------------------------------------
##Objective Function (Model B)
#-------------------------------------------------------------------------------

@objective( modelB, Min, sum(TPL[p,l]*(CV[g]*q[g,p,l] + CF[g]*u[g,p,l]) for g in G, p in P, l in L) )
#print(model)
status = solve(modelB)

println("Objective value: ", getobjectivevalue(modelB))
println("q = ", getvalue(q))
println("u = ", getvalue(u))

R_gen_B = zeros(length(n_Data[:,1]),G[end])
for g in G
	for i in 1:length(n_Data[:,1])
		R_gen_B[i,g] = getvalue(q)[g,:,n_Data[i,3]][1]
	end
end

R_dis_B = zeros(length(n_Data[:,1]),G[end])
for g in G
	for i in 1:length(n_Data[:,1])
		R_dis_B[i,g] = getvalue(u)[g,:,n_Data[i,3]][1]
	end
end


#R_gen_B
#R_dis_B

CSV.write("out1_B.csv",convert(DataFrame,[R_gen_B R_dis_B]))
#CSV.write("out2_A.csv",convert(DataFrame,[R_up_A R_down_A]))



#-------------------------------------------------------------------------------
# Model A formulation
#-------------------------------------------------------------------------------

modelA = Model(solver = GurobiSolver())

#-------------------------------------------------------------------------------
##Sets (Model A)
#-------------------------------------------------------------------------------

G = collect(1:13)           # Thermal Generators
#                           # 1:     Nuclear
#                           # 2-6:   Coal
#                           # 7-11:  Combined Cycle
#                           # 12-13: Fuel oil
P = collect(1:168)            # Number of hours (1 week) = 7*24

#-------------------------------------------------------------------------------
##Parameters (Model A)
#-------------------------------------------------------------------------------
VP = W
DP = D
FP = 0.4*DP

#-------------------------------------------------------------------------------
##Variables (Model A)
#-------------------------------------------------------------------------------

@variable(modelA,q[1:G[end], 1:P[end]] >= 0)
@variable(modelA,0 <= v[1:P[end]] <= 2000)
@variable(modelA,u[1:G[end], 1:P[end]], Bin)
@variable(modelA,0 <= y[1:G[end], 1:(P[end]-1)] <= 1)
@variable(modelA,0 <= z[1:G[end], 1:(P[end]-1)] <= 1)

#-------------------------------------------------------------------------------
##Constraints (Model A)
#-------------------------------------------------------------------------------

@constraint(modelA,[g in G, p in P], q[g,p] <= Qmax[g]*u[g,p])
@constraint(modelA,[g in G, p in P], q[g,p] >= Qmin[g]*u[g,p])
@constraint(modelA,[p in P], v[p] <= VP[p])
@constraint(modelA,[g in G,p in P[1:(P[end]-1)]], u[g,p + 1] - u[g,p] == y[g,p] - z[g,p])
@constraint(modelA,[p in P], sum(q[g,p] for g in G) == (DP[p] - v[p]))
@constraint(modelA,[p in P], FP[p] <= sum((Qmax[g]*u[g,p] - q[g,p]) for g in G))

#-------------------------------------------------------------------------------
##Objective Function (Model A)
#-------------------------------------------------------------------------------

@objective( modelA, Min, sum(CV[g]*q[g,p] + CF[g]*u[g,p] for g in G, p in P) +
                        sum(C_up[g]*y[g,p] + C_dn[g]*z[g,p] for g in G, p in P[1:(P[end]-1)]) )
#print(model)
status = solve(modelA)

println("Objective value: ", getobjectivevalue(modelA))
println("q = ", getvalue(q))
println("v = ", getvalue(v))
println("u = ", getvalue(u))
println("y = ", getvalue(y))
println("z = ", getvalue(z))

R_gen_A = zeros(G[end],P[end])
for g in G
	for p in P
		R_gen_A[g,p] = getvalue(q)[g,p][1]
	end
end

R_win_A = zeros(P[end])
for p in P
	R_win_A[p] = getvalue(v)[p][1]
end

R_dis_A = zeros(G[end],P[end])
for g in G
	for p in P
		R_dis_A[g,p] = getvalue(u)[g,p][1]
	end
end

R_up_A = zeros(P[end]-1,G[end])
for g in G
	for p in P[1:(P[end]-1)]
		R_up_A[p,g] = getvalue(y)[g,p][1]
	end
end

R_down_A = zeros(P[end]-1,G[end])
for g in G
	for p in P[1:(P[end]-1)]
		R_down_A[p,g] = getvalue(z)[g,p][1]
	end
end

#R_gen_A
#R_dis_A
#R_win_A
#R_up_A
#R_down_A

#sp = W - R_win_A

#CSV.write("output_gen.csv",convert(DataFrame,R_gen_A))
#CSV.write("output_win.csv",convert(DataFrame,[R_win_A zeros(size(R_win_A))]))

CSV.write("out1_A.csv",convert(DataFrame,[transpose(R_gen_A) transpose(R_dis_A) R_win_A]))
CSV.write("out2_A.csv",convert(DataFrame,[R_up_A R_down_A]))


#CV
#CF
