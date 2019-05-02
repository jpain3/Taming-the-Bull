#BEER GAME OPTIMIZATION TOOL
#James Paine
#jpaine@mit.edu

#Set the working directory
setwd("C:/Users/jpaine/OneDrive/MIT/15.071 - Analytics Edge/Project")

#Load the support function that acutally runs the simulation game for a given timestep and action/state pair
source("PirateBeerGame_Opt_Function.R")

###FUNCTIONS####

  #Helper function to write output to excel
  write.excel <- function(x,row.names=FALSE,col.names=TRUE,...) {
    write.table(x,"clipboard",sep="\t",row.names=row.names,col.names=col.names,...)
  }
  
  
  #Function to reset the size of the game space
  reset_game <- function(horizon = 36, Initial_Inventory = 12, Information_Delay = 2, Shipping_Delay = 2) {
  
    # Initial_Inventory = 12
    # horizon = 102
    # Information_Delay = 2
    # Shipping_Delay = 2
    
    #Customer Order String
    #Classic Beer Game
    Step_Round = 5
    Orders = append(rep(4,Step_Round),rep(8,((horizon+1)-Step_Round)))
    
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
  
  # Reward = "Cost" #value of 'Cost' or 'Amp'
  # fnscale = 1 #Set to 1 to minimize the above reward, or set to -1 to maximize the above reward
  
  Reward = "Amp"
  fnscale = -1
  
  Master_Start = Sys.time()
  Opt = list()
  
  #Reset the environment and set the horizon for the optimization
  horizon = 36
  Reset_List = reset_game(horizon = horizon)
  list2env(Reset_List, envir = .GlobalEnv)
  
  Parameter_df = data.frame(theta = rep(0.36,4),
                            alpha_s = rep(0.26,4),
                            beta = rep(0.34,4),
                            S_prime = rep(17,4))
  
  for (ent in 1:4) {
    
    start = Sys.time()
    
    print(paste("Optimizing over entity",ent,sep=" "))
    
    Initial_guess = c(0.36,0.26,0.34,17)
    
    AI_Entity_Index = ent
    
    #Change the method in the below function to explore different optimization results
    Opt_output = optim(Initial_guess, BeerGame_Optimization, control=list(fnscale=fnscale), 
                       AI_Entity_Index = AI_Entity_Index, horizon = horizon, Reward = Reward, Parameter_df = Parameter_df,
                       #method = c("CG"),
                       #method = c("Nelder-Mead"),
                       #lower = c(0,0,0,0) #, upper = c(2,1,5,100)
                       )
    
    end = Sys.time()
    time_to_optimize = end - start
    
    print(paste("Done in",round(time_to_optimize,2),"seconds"))
    print(paste(c("theta =","alpha_s =","beta =","S_prime ="), Opt_output$par))
    print(paste("Function Value = ",Opt_output$value))
    
    name <- as.character(ent)
    
    Opt[[name]] = Opt_output
  
  } # next entity
  
  paste("Optimization Done in",round((Sys.time()-Master_Start),2),"seconds!")
  
  #Output the optimized parameters to the clipboard
  Opt_Par = t(data.frame(Opt$`1`$par,Opt$`2`$par,Opt$`3`$par,Opt$`4`$par))
  rownames(Opt_Par) = c("1","2","3","4")
  colnames(Opt_Par) = c("theta","alpha_s","beta","S_prime")
  
  Opt_Par
  
  write.excel(Opt_Par)
  ###############################



#####SEQUENTIAL GREEDY OPTIMIZATION#####
  # This runs the optimization in sequence, keeping the previously optimized values for lower number entities
  #   in the supply chain. Eg, Entity 1 is optimized versus 'average typical humans' in positions 2,3, and 4
  # Next, entity 2 is optimized versus the optimzied parameters found for entity 1 and 'average typical humans'
  #   in positions 3 and 4, etc.
  
  
  Reward = "Cost" #value of 'Cost' or 'Amp'
  fnscale = 1 #Set to 1 to minimize the above reward, or set to -1 to maximize the above reward
  
  
  Master_Start = Sys.time()
  Opt_seq = list()
  
  #Reset the environment and set the horizon for the optimization
  horizon = 104
  Reset_List = reset_game(horizon = horizon)
  list2env(Reset_List, envir = .GlobalEnv)
  
  Parameter_df = data.frame(theta = rep(0.36,4),
                            alpha_s = rep(0.26,4),
                            beta = rep(0.34,4),
                            S_prime = rep(17,4))
  
  for (ent in 1:4) {
    
    start = Sys.time()
    
    print(paste("Optimizing over entity",ent,sep=" "))
    
    Initial_guess = c(0.36,0.26,0.34,17)
    
    AI_Entity_Index = ent
    
    print(Parameter_df)
    
    #Change the method in the below function to explore different optimization results
    Opt_output = optim(Initial_guess, BeerGame_Optimization, control=list(fnscale=fnscale), 
                       AI_Entity_Index = AI_Entity_Index, horizon = horizon, Reward = Reward, Parameter_df = Parameter_df,
                       method = c("CG")
                       #method = c("Nelder-Mead")
                       #lower = 0
                        )
    
    end = Sys.time()
    time_to_optimize = end - start
    
    print(paste("Done in",round(time_to_optimize,2),"seconds"))
    print(paste(c("theta =","alpha_s =","beta =","S_prime ="), Opt_output$par))
    print(paste("Function Value = ",Opt_output$value))
    
    name <- as.character(ent)
    
    Opt_seq[[name]] = Opt_output
    
    Parameter_df[ent,] = Opt_output$par
    
  } # next entity
  
  paste("Optimization Done in",round((Sys.time()-Master_Start),2),"seconds!")
  
  #Output the combined optimizatoin
  Parameter_df
  
  #Write the results to the clipboard
  Opt_Par_seq = t(data.frame(Opt_seq$`1`$par,Opt_seq$`2`$par,Opt_seq$`3`$par,Opt_seq$`4`$par))
  rownames(Opt_Par_seq) = c("1","2","3","4")
  colnames(Opt_Par_seq) = c("theta","alpha_s","beta","S_prime")
  
  Opt_Par_seq
  
  write.excel(Opt_Par_seq)
    
  ###############################


#####ITERATIVE SEQUENTIAL GREEDY OPTIMIZATION####
  # This runs the optimization many times, seeking convergence of results. Each iteration, the entities are optimized in isolation
  #   using the 'average typical human' player behavior for the other entities for the first iteration. Thereafter, the behavior
  #   parameters are updated to the optimzied values, and the process repeats. For all optimization methods, convergence is achieved
  #   within 10 to 15 iterations.
  
  Reward = "Cost" #value of 'Cost' or 'Amp'
  fnscale = 1 #Set to 1 to minimize the above reward, or set to -1 to maximize the above reward
  
  Master_Start = Sys.time()
  Opt_seq = list()
  
  #Set the maximum number of iterations
  Num_Iterations = 20
  
  #Reset the environment and set the horizon for the optimization
  horizon = 104
  Reset_List = reset_game(horizon = horizon)
  list2env(Reset_List, envir = .GlobalEnv)
  
  #Set the initialized conditions to optimize over
  Parameter_df = data.frame(theta = rep(0.36,4),
                            alpha_s = rep(0.26,4),
                            beta = rep(0.34,4),
                            S_prime = rep(17,4))
  
  #Reset the dataframe that holds the results
  Parameter_df_2 = Parameter_df
  
  for (r in 1:Num_Iterations) {
  
    print(paste("Iteration",r,"of",Num_Iterations))
    
    Parameter_df = Parameter_df_2
    
    for (ent in 1:4) {
      
      start = Sys.time()
      
      print(paste("Optimizing over entity",ent,sep=" "))
      
      Initial_guess = unlist(unname(Parameter_df[ent,]))
      
      AI_Entity_Index = ent
      
      #Change the method in the below function to explore different optimization results
      Opt_output = optim(Initial_guess, BeerGame_Optimization, control=list(fnscale=fnscale), 
                         AI_Entity_Index = AI_Entity_Index, horizon = horizon, Reward = Reward, Parameter_df = Parameter_df,
                         method = c("BFGS"),
                         #method = c("SANN")
                         #method = c("CG")
                         #method = c("Nelder-Mead")
                         lower = c(0,0,0,0)
      )
      
      end = Sys.time()
      time_to_optimize = end - start
      
      #print(paste("Done in",round(time_to_optimize,2),"seconds"))
      #print(paste(c("theta =","alpha_s =","beta =","S_prime ="), Opt_output$par))
      #print(paste("Function Value = ",Opt_output$value))
      
      name <- as.character(ent)
      
      Opt_seq[[name]] = Opt_output
      
      Parameter_df_2[ent,] = Opt_output$par
      
    } # next entity
    
    print(Parameter_df_2)
    print(paste("   Optimization Function Value =",Opt_output$value))
    print(paste("   Optimization Done in",round((Sys.time()-Master_Start),2),"seconds!"))
  
  } #next iteration
  
  #Write the final optimized parameters to the clipboard  
  Opt_Par_seq = t(data.frame(Opt_seq$`1`$par,Opt_seq$`2`$par,Opt_seq$`3`$par,Opt_seq$`4`$par))
  rownames(Opt_Par_seq) = c("1","2","3","4")
  colnames(Opt_Par_seq) = c("theta","alpha_s","beta","S_prime")
  Opt_Par_seq
  write.excel(Opt_Par_seq)
  ###############################
  