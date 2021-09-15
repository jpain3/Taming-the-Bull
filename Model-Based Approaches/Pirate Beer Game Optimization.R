#BEER GAME OPTIMIZATION TOOL
#James Paine
#jpaine@mit.edu

#https://www.rdocumentation.org/packages/optimx/versions/2020-2.2/topics/optimx

#Set the working directory
#setwd("C:/Users/jpaine\OneDrive/MIT/Beer Game Project/Rogelio updates")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Load the support function that acutally runs the simulation game for a given timestep and action/state pair
source("PirateBeerGame_Opt_Function_Backorderfix.R")
require("optimx")
#require("Rcgmin")

###FUNCTIONS####

#Helper function to write output to excel
write.excel <- function(x,row.names=FALSE,col.names=TRUE,...) {
  write.table(x,"clipboard",sep="\t",row.names=row.names,col.names=col.names,...)
}

#Helper function to write percents
percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

#Function to reset the size of the game space
reset_game <- function(horizon = 36, Initial_Inventory = 12, Information_Delay = 2, Shipping_Delay = 2, Random_Orders = FALSE) {
  
  # Initial_Inventory = 12
  # horizon = 102
  # Information_Delay = 2
  # Shipping_Delay = 2
  if (Random_Orders == FALSE) {
    #Customer Order String
    #Classic Beer Game
    Step_Round = 5
    Orders = append(rep(4,Step_Round),rep(8,((horizon+1)-Step_Round)))
  } else {
    
    set.seed(123)
    
    #Normally Distributed demand
    Order_mean = 6
    Order_SD = 2
    Orders = rnorm(horizon+1,Order_mean, Order_SD)
    
  }
  
  ##################
  #Setting up the game
  ##################
  
  
  Order_flows = array(NA,dim = c(4,Information_Delay,(horizon+1)))
  #populate initial values of orders
  Order_flows[,,1] = Orders[1]
  
  Shipping_flows = array(NA,dim = c(4,Shipping_Delay,(horizon+1)))
  #populate initial values of incomming shippments
  Shipping_flows[,,1] = Orders[1]
  
  OH_Inventory = array(NA,dim=c(4,(horizon+1)))
  OH_Inventory[,1] = Initial_Inventory
  
  Backorder = array(NA,dim=c(4,(horizon+1)))
  Backorder[,1] = 0
  
  L_hat = array(NA,dim=c(4,(horizon+1)))
  L = array(NA,dim=c(4,(horizon+1)))
  
  Final_Customer_Orders_Filled = rep(NA, (horizon+1))
  Production_Request = rep(NA,(horizon+1))
  Production_Request[1] = Orders[1]
  
  Amp_Vector = rep(NA,horizon+1)
  Reward_Vector = rep(NA,horizon+1)
  
  Output <- list(Orders,Order_flows,Shipping_flows,OH_Inventory,Backorder,L_hat,L,
                 Final_Customer_Orders_Filled,Production_Request,
                 Amp_Vector, Reward_Vector)
  
  names(Output) = c("Orders", "Order_flows", "Shipping_flows", "OH_Inventory", "Backorder", "L_hat", "L",
                    "Final_Customer_Orders_Filled", "Production_Request",
                    "Amp_Vector", "Reward_Vector")
  
  return(Output)
  
}



#FUNCTION TO ITERATE OVER THE TIME HORIZON FOR THE GIVEN PARAMETERS
#FUNCTION TO OPTIMIZE
BeerGame_Optimization <- function(Agent_Parameters, AI_Entity_Index = 3, horizon = 36, Parameter_df = FALSE,
                                  Holding_Cost = 0.5, Backorder_Cost = 1, Reward = "Amp") {
  
  Parameter_df = FALSE
  AI_parameter_df = Agent_Parameters
  names(AI_parameter_df) = c("theta","alpha_s","beta","S_prime")
  AI_Entity = TRUE
  AI_Order = FALSE
  
  AI_Entity_Index = AI_Entity_Index
  horizon = horizon
  
  for (t in 2:(horizon+1)) {
    
    #NESTED FUNCTION TO RUN THE GAME FOR ONE TIME STEP AND RETURN THE NEW STATE
    R_function_output = PirateBeerGame_Opt(AI_Entity, AI_Entity_Index = AI_Entity_Index, AI_parameter_df = AI_parameter_df, AI_Order = AI_Order, 
                                           t = t, Orders = Orders,
                                           Order_flows = Order_flows, Shipping_flows = Shipping_flows, OH_Inventory = OH_Inventory, Backorder = Backorder,
                                           L = L, L_hat = L_hat, Production_Request = Production_Request,
                                           Parameter_df = Parameter_df)
    
    #Orders = R_function_output$Orders
    Order_flows = R_function_output$Order_flows
    Shipping_flows = R_function_output$Shipping_flows
    OH_Inventory = R_function_output$OH_Inventory
    Backorder = R_function_output$Backorder
    L = R_function_output$L
    L_hat = R_function_output$L_hat
    Production_Request = R_function_output$Production_Request
    Reward_Vector[t]=R_function_output$reward
    
  } #next t
  
  #Calculate costs
  Costs_Per_Period = OH_Inventory*Holding_Cost + Backorder*Backorder_Cost
  Cummulative_Costs_Per_Period = t(apply( Costs_Per_Period ,1 , cumsum))
  Total_Costs_Per_Entity = rowSums(Costs_Per_Period)
  Total_Team_Costs = sum(Total_Costs_Per_Entity)
  
  #Calculate reward based on amplitude function
  total_reward = sum(Reward_Vector, na.rm=TRUE)
  
  if (Reward == "Amp") {
    return(total_reward)
  } else {
    return(Total_Team_Costs)
  }
  
  
} #End optimization function


###OPTIMIZATION TYPES##########

###MASTER OPTIMIZATION#########
# This runs the optimization for each entity position in sequence, assuming all other positions act as
# 'average human' players.

Reward = "Cost" #value of 'Cost' or 'Amp'
fnscale = 1 #Set to 1 to minimize the above reward, or set to -1 to maximize the above reward

Reward = "Amp"
fnscale = 1

opt_list = c("L-BFGS-B","Rcgmin","bobyqa")

Master_Start = Sys.time()
Opt = list()

#Reset the environment and set the horizon for the optimization
horizon = 36
Shipping_Delay = 2
Initial_Inventory = 12
axial_improvement_search = TRUE

Stable_Start = TRUE
Shipping_Delay = 2

Parameter_df = data.frame(theta = rep(0.36,4),
                          alpha_s = rep(0.26,4),
                          beta = rep(0.34,4),
                          S_prime = rep(17,4))

#Enforce stable start if that flag is present
if (Stable_Start == TRUE) {
  
  #Assign parameter values
  theta = unlist(Parameter_df$theta)
  alpha_s = unlist(Parameter_df$alpha_s)
  beta = unlist(Parameter_df$beta)
  S_prime = unlist(Parameter_df$S_prime)
  
  Initial_Supply_Line = Shipping_Delay*4
  Initial_Inventory = S_prime - beta*Initial_Supply_Line
  
}


Reset_List = reset_game(Shipping_Delay = Shipping_Delay, Initial_Inventory = Initial_Inventory, horizon = horizon)
list2env(Reset_List, envir = .GlobalEnv)

for (ent in 1:4) {
  
  start = Sys.time()
  
  print(paste("Optimizing over entity",ent,sep=" "))
  
  Initial_guess = c(0.36,0.26,0.34,17)
  Initial_guess = c(0.36,0.26,1,36)
  #Initial_guess = c(0.36,0.26,0.8,35)
  #Initial_guess = c(0.5,0.5,0.5,36)
  
  AI_Entity_Index = ent
  
  #AI_Entity_Index=1
  
  #Change the method in the below function to explore different optimization results
  Opt_output = optimx(par = Initial_guess, fn = BeerGame_Optimization, control=list(fnscale=fnscale), 
                      AI_Entity_Index = AI_Entity_Index, horizon = horizon, Reward = Reward, Parameter_df = Parameter_df,
                      #method = "Nelder-Mead"
                      method = "L-BFGS-B", lower = c(0,0,0,0) , upper = c(1,1,1,Inf)
                      #method = "Rcgmin", lower = c(0,0,0,0) , upper = c(1,1,1,Inf)
                      #method = "bobyqa", lower = c(0,0,0,0) , upper = c(1,1,1,Inf)
                      )
  
  end = Sys.time()
  time_to_optimize = end - start
  
  Opt_par = unname(unlist(Opt_output[1:4]))
  
  print(paste("Done in",round(time_to_optimize,2),"seconds"))
  print(paste(c("theta =","alpha_s =","beta =","S_prime ="), Opt_par))
  print(paste("Function Value = ",Opt_output$value))
  
  name <- as.character(ent)
  
  Opt[[name]] = Opt_par
  
  if (axial_improvement_search==TRUE){
    print("Axial Searching for Improvements")
    
    axial_search = axsearch(par = unname(Opt_par), fn = BeerGame_Optimization, fmin = Opt_output$value,
                             lower = c(0,0,0,0) , upper = c(1,1,1,Inf),
                             AI_Entity_Index = AI_Entity_Index, horizon = horizon, Reward = Reward, Parameter_df = Parameter_df)
    
    if (axial_search$bestfn<Opt_output$value) {
      improvement = percent((axial_search$bestfn-Opt_output$value)/Opt_output$value,digits=8)
      print(paste("Improvement of ",improvement,"... Saving improved values..."),sep="")
      Opt[[name]] = axial_search$par
    } else {
      print("No improvement found.")
    }
  }
  
} # next entity

paste("Optimization Done in",round((Sys.time()-Master_Start),2),"seconds!")

#Output the optimized parameters to the clipboard
Opt_Par = t(data.frame(Opt$`1`,Opt$`2`,Opt$`3`,Opt$`4`))
rownames(Opt_Par) = c("1","2","3","4")
colnames(Opt_Par) = c("theta","alpha_s","beta","S_prime")

Opt_name = rownames(Opt_output)

Opt_name
Opt_Par

#write.excel(Opt_Par)
###############################


###Run Game with Optimized Values and Summarize#########


#Reset the environment and set the horizon for the optimization
horizon = 104
Random_Orders = FALSE
#Random_Orders = TRUE


Parameter_df = data.frame(theta = rep(0.36,4),
                          alpha_s = rep(0.26,4),
                          beta = rep(0.34,4),
                          S_prime = rep(17,4))

#Enforce stable start if that flag is present
if (Stable_Start == TRUE) {
  
  
  #Assign parameter values
  theta = unlist(Parameter_df$theta)
  alpha_s = unlist(Parameter_df$alpha_s)
  beta = unlist(Parameter_df$beta)
  S_prime = unlist(Parameter_df$S_prime)
  
  Initial_Supply_Line = Shipping_Delay*4
  Initial_Inventory = S_prime - beta*Initial_Supply_Line
  
}


Reset_List = reset_game(Shipping_Delay = Shipping_Delay, Initial_Inventory = Initial_Inventory, horizon = horizon)
list2env(Reset_List, envir = .GlobalEnv)


Total_Cost_Amp_Based = vector()
Total_Cost_Inventory_Based = vector()

for (i in 1:4) {

  Entity_Index = i
  
  Parameter_df = Parameter_df_orig
  Parameter_df[Entity_Index,]=Opt_Par[Entity_Index,]
  
  #Uncomment below to generate 'baseline' costs
  #Parameter_df = Parameter_df_orig
  
  AI_Entity = FALSE
  AI_Order = FALSE
  Information_Delay = 2

  for (t in 2:(horizon+1)) {
    
    #NESTED FUNCTION TO RUN THE GAME FOR ONE TIME STEP AND RETURN THE NEW STATE
    R_function_output = PirateBeerGame_Opt(t = t, Orders = Orders,
                                           Order_flows = Order_flows, Shipping_flows = Shipping_flows, OH_Inventory = OH_Inventory, Backorder = Backorder,
                                           L = L, L_hat = L_hat, Production_Request = Production_Request,
                                           Parameter_df = Parameter_df)
    
    #Orders = R_function_output$Orders
    Order_flows = R_function_output$Order_flows
    Shipping_flows = R_function_output$Shipping_flows
    OH_Inventory = R_function_output$OH_Inventory
    Backorder = R_function_output$Backorder
    L = R_function_output$L
    L_hat = R_function_output$L_hat
    Production_Request = R_function_output$Production_Request
    Reward_Vector[t]=R_function_output$reward
    
    Final_Orders = unname(rbind(Order_flows[2:4,Information_Delay,],Production_Request))
    
  } #next t
  
  #Calculate costs
  Holding_Cost = 0.5
  Backorder_Cost = 1
  
  Amplification = (abs(Final_Orders-Orders))/Orders
  TotalAmpCosts = abs(sum(-25*Amplification^2 + 1))
  
  Costs_Per_Period = OH_Inventory*Holding_Cost + Backorder*Backorder_Cost
  Cummulative_Costs_Per_Period = t(apply( Costs_Per_Period ,1 , cumsum))
  Total_Costs_Per_Entity = rowSums(Costs_Per_Period)
  Total_Team_Costs = sum(Total_Costs_Per_Entity)
  
  #Calculate reward based on amplitude function
  total_amp_reward = sum(Reward_Vector, na.rm=TRUE)
  
  Total_Cost_Amp_Based[i]=TotalAmpCosts
  Total_Cost_Inventory_Based[i]=Total_Team_Costs

}

Extended_Opt_Par = cbind(Opt_Par,Total_Cost_Inventory_Based,Total_Cost_Amp_Based)

rownames(Extended_Opt_Par)=paste(toupper(Reward),Opt_name,rownames(Opt_Par),sep=" ")

Reward
Opt_name
Extended_Opt_Par

write.excel(Extended_Opt_Par,row.names=TRUE)
Reward
Opt_name
Extended_Opt_Par