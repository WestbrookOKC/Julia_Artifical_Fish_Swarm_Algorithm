using CSV, DataFrames,Flux,Plots,Random,Distributions
using Flux:onehot
using Statistics
using Printf
real_money = DataFrame(CSV.File("./Data/real-money-traindata.csv",normalizenames=true))#533*5
fack_money =DataFrame(CSV.File("./Data/notgood-traindata.csv",normalizenames=true))#427*5
x_real_money = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(real_money) ]#533
x_fake_money = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(fack_money) ]#427
xs = vcat(x_real_money,x_fake_money) # change the structure of it
ys = vcat( fill(onehot(0,0:1), size(x_real_money)),
           fill(onehot(1,0:1), size(x_fake_money)))  # make onehot to them
Actual = [fill(0,size(x_real_money)); fill(1,size(x_fake_money))] # Store the correct result
#prediction(i) = findmax(model(Flux.batch(xs[i])))[2] - 1
# It shows Confusion Matrix of a model
function Confusion_Martix!(model)
    TP,FP,TN,FN = 0,0,0,0
    for i in 1:960
        predict_value = findmax(model(Flux.batch(xs[i])))[2] - 1
        if predict_value == 0 && Actual[i] ==  0
            TP += 1
        end
        if predict_value == 0 && Actual[i] ==  1
            FP += 1
        end
        if predict_value == 1 && Actual[i] ==  1
            TN += 1
        end
        if predict_value == 1 && Actual[i] ==  0
            FN += 1
        end
    end
    print([TP FN
            FP TN])
end
# Create a model here
function getmodel()
    model = Chain(Dense(4,8,σ),Dense(8,2,identity),softmax)
end
# It generate fishes according to size
function population(size)
    list = []
    for i in 1:size
        rand_model = getmodel()
        append!(list,[rand_model])
    end
    return list
end
# Get the accuracy of a single model
function accuracy_each_model(rand_model)
    Actual = [fill(0,size(x_real_money)); fill(1,size(x_fake_money))]
    prediction(i) = findmax(rand_model(xs[i]))[2] - 1
    TP,FP,TN,FN = 0,0,0,0
    for i in 1:960
        predict_value = prediction(i)
        if predict_value == 0 && Actual[i] ==  0
            TP += 1
        end
        if predict_value == 0 && Actual[i] ==  1
            FP += 1
        end
        if predict_value == 1 && Actual[i] ==  1
            TN += 1
        end
        if predict_value == 1 && Actual[i] ==  0
            FN += 1
        end
    end
    accuracy = (TN + TP) / (TP+TN+FN+FP)
    return accuracy
end

#get accuracy list and use dict to link them
function get_acc_list(model_list)
    acc_list = []
    for acc in model_list
        append!(acc_list,accuracy_each_model(acc))
    end
    acc_list = sort(acc_list,rev=true)
    for acc in model_list
        key = accuracy_each_model(acc)
        value = acc
        if dict.keys != key
        push!(dict,key =>value)
        end
        if dict.keys == key
            dict[key] = acc
        end
    end
    return acc_list
end
# update global best
function bulletin(acc_list) 
    acc_list = sort(acc_list,rev = true)
    return acc_list[1]
end
# It is the Prey function, do the movement based on the following statements.
#X_next[i] = X[i] + visual *rand()
#X_next[i] = X[i] + (X_next[i] - X[i])/abs(X_next[i]-X[i])*step*rand()
function prey(model)
    i = 0
    while i < trynum # If the prey cost too much time, you can decrease the trynum.
        i = i + 1
        a = model # create a temp model
        rander = rand([-1,1])*rand()
        #update temp model layer1's weight
        for j in 1:32
            a[1].weight[j] = a[1].weight[j] + visual*rander
        end
        #judge if accuracy got improved
        if accuracy_each_model(a) > accuracy_each_model(model)
            for j in 1:32
                X_next = model[1].weight[j] +viusal*rander
                model[1].weight[j] = model[1].weight[j] + (X_next - model[1].weight[j])/abs(X_next - model[1].weight[j])*step*rander
            end
            
            break
        end
        #for layer2
        for k in 1:16
            a[2].weight[k] = a[2].weight[k] + visual*rander
        end
        if accuracy_each_model(a) > accuracy_each_model(model)
            for k in 1:16
                X_next = model[2].weight[k] +viusal*rander
                model[2].weight[k] = model[2].weight[k] + (X_next - model[2].weight[k])/abs(X_next - model[2].weight[k])*step*rander
            end
            break
        end
    end
    for j in 1:32
        model[1].weight[j] = model[1].weight[j] + visual*rand([-1,1])*rand()
    end
    for k in 1:16
        model[2].weight[k] = model[2].weight[k] + visual*rand([-1,1])*rand()
    end
    #print("not found:")
    #print(accuracy_each_model(model))
end

function evaluate_swarm(model,idx)# get center of neighbour
    neighbour = []
    nf = 0
    #model = fishes
    for i in 1:100 # get neighbour for layer1's weight[1]
        if abs(fishes[i][1].weight[idx] - model[1].weight[idx]) < visual
            append!(neighbour,fishes[i][1].weight[idx])
            nf = nf + 1
        end    
    end
    center = sum(neighbour) / nf
    return center,nf/100
end
    # k={Xj|Xj-Xi <= visual} nf is neighbour
    # nf / n should smaller than delta
    # if Yc > Yi && nf / n < delta
    # X_next[i] = X[i] + (X_next[i] - X[i])/abs(X_next[i]-X[i])*step*rand()
    # Xc = (x1+x2+...xn) / n
function swarm(model)
    for i in 1:32 #swarm for layer1
        eval = evaluate_swarm(model,i)
        center = eval[1]
        nf = eval[2]
        temp = model
        temp[1].weight[i] = center
        if accuracy_each_model(temp) > accuracy_each_model(model) && nf < delta# if Yc>Yi and nf < delta 
            model[1].weight[i] = model[1].weight[i] + (temp[1].weight[i] - model[1].weight[i])/abs(temp[1].weight[i] - model[1].weight[i])*step*rand([-1,1])*rand()
            print(accuracy_each_model(model))
        else # too crowd or no good acc
            prey(model)
        end
    end
end
    
    # distanceij < visual
    # calculate neighbour's accuracy
    # if neighbour's accuracy better, move to it
    # X_next[i] = X[i] + (X_next[i] - X[i])/abs(X_next[i]-X[i])*step*rand()
function follow(model)
    
    for idx in 1:32
        neighbour = []
        ids = []
        nf = 0
        #model = fishes
        for i in 1:100 # get neighbour
            if abs(fishes[i][1].weight[idx] - model[1].weight[idx]) < visual
                append!(neighbour,fishes[i][1].weight[idx])
                append!(ids,i)
                nf = nf + 1
            end    
        end
        nf = nf /100
        for id in ids #follow the good neighbour
            if accuracy_each_model(fishes[id]) > accuracy_each_model(model) && nf < delta
                model[1].weight[idx] = model[1].weight[idx] + (fishes[id][1].weight[idx]-model[1].weight[idx])/abs(fishes[id][1].weight[idx]-model[1].weight[idx])*step*rand([-1,1])*rand()
            end
        end
    end
    #for layer2
    for idx in 1:16
        neighbour = []
        ids = []
        nf = 0
        #model = fishes
        for i in 1:100 # get neighbour
            if abs(fishes[i][2].weight[idx] - model[2].weight[idx]) < visual
                append!(neighbour,fishes[i][2].weight[idx])
                append!(ids,i)
                nf = nf + 1
            end    
        end
        nf = nf /100
        for id in ids #follow the good neighbour
            if accuracy_each_model(fishes[id]) > accuracy_each_model(model) && nf < delta
                model[2].weight[idx] = model[2].weight[idx] + (fishes[id][2].weight[idx]-model[2].weight[idx])/abs(fishes[id][2].weight[idx]-model[2].weight[idx])*step*rand([-1,1])*rand()
            end
        end
    end
end




#main part of Julia

fishnum = 100        # The population size
visual = 0.5         # How far a fish can see
delta = 0.05         # Crowding factor
step = 0.05          # How far a fish can move
trynum = 50          # The maximum try time in Prey function
itermax = 100        # The maximum number of generations
dict = Dict()        # It links the accuracy and model

fishes = population(100)        # generate 100 model list
acc_list = get_acc_list(fishes) # save accuracy
final_acc = []                  # It is used to test that how many iterations it cost to get best solution.

# The whole process, it will break while bulletin bigger than 90%.
for i in 1:itermax
    acc_list = get_acc_list(fishes)
    follow(dict[acc_list[i]])
    swarm(dict[acc_list[i]])
    acc_list = get_acc_list(fishes)
    append!(final_acc,acc_list[1])
    if bulletin(acc_list) > 0.8
        break
    end
end

function show_accuracy()
    Plots.plot(1:length(acc_list), acc_list, label = "AFSA")
end
Confusion_Martix!(dict[acc_list[1]])# it shows the confusion martix of the best individual.
acc_list = sort(acc_list)
show_accuracy() # it shows the generation(model list) accuracy.

temp = [] # use to contain the original weight distribution
for i in 1:100
    append!(temp,fishes[i][1].weight[1])
end
scatter(temp,1:length(temp),c=:orange) # shows distribution of original weight
# after AFSA use it
temp2 = [] # use to contain the disribution of weight, after doing AFSA.
for i in 1:100
    append!(temp2,fishes[i][1].weight[1])
end
scatter(temp2,1:length(temp2))# shows distribution of  weight
