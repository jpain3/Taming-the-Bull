import csv
import time
import datetime
import random
import copy
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete, Box

def PirateBeerGame_funct(AI_Entity_Index, AI_Order, Orders, Order_flows, Shipping_flows, OH_Inventory,
                         Backorder, L_hat, Production_Request, AI_Entity=False, AI_parameter_df=False,
                         Integer_Ordering=False, Noisey_Ordering=False, Noise_Mean=0, Noise_SD=1,
                         Restricted_Ordering = False, Restricted_Order = 4,
                         Parameter_df=False, Shipping_Delay=2, Information_Delay=2):
    ##DEBUG ONLY
    # Shipping_Delay = 2
    # Information_Delay = 2
    # AI_Entity = False
    # AI_Entity_Index = 3
    # t = 0
    # OH_Inventory = [12]*4
    # Shipping_flows = [[4]*2]*4
    # Order_flows = [[4]*2]*4
    # Backorder = [0]*4
    # Orders = 4
    # Production_Request = 4
    # L_hat = [4.00]*4

    Final_Orders = np.empty(4, dtype=float)
    OH_Inventory = np.array(OH_Inventory)
    Shipping_flows = np.array(Shipping_flows)
    Order_flows = np.array(Order_flows)

    # Ensure that the order flow facing the retailer is the actual customer order
    Order_flows[0, 0] = Orders

    # Read in the ordering paramters
    if Parameter_df != False:
        theta = Parameter_df['theta']
        alpha_s = Parameter_df['alpha_s']
        beta = Parameter_df['beta']
        S_prime = Parameter_df['S_prime']
    else:
        theta = [0.36] * 4
        alpha_s = [0.26] * 4
        beta = [0.34] * 4
        S_prime = [17] * 4

        TeamName = "Default Average Agents"

    # Read in AI Ordering Parameters if present
    if AI_parameter_df != False:
        theta[AI_Entity_Index] = AI_parameter_df['theta']
        alpha_s[AI_Entity_Index] = AI_parameter_df['alpha_s']
        beta[AI_Entity_Index] = AI_parameter_df['beta']
        S_prime[AI_Entity_Index] = AI_parameter_df['S_prime']

    #####Recieve Inventory and Advance Shipping Delays#####

    # Recieve shipments
    New_OH_Inventory = OH_Inventory + Shipping_flows[:, 0]

    # Advance shippping delays
    Shipping_flows[:, 0] = Shipping_flows[:, (Shipping_Delay - 1)]
    #Shipping_flows[:, (Shipping_Delay - 1)] = np.nan

    #####Fill Orders######

    # View Orders
    Order_Received = Order_flows[:, 0]
    # Calculate net order that needs to be fullfilled
    Incoming_Order = Order_flows[:, 0] + Backorder
    # Ship what you can
    Outbound_shipments = np.maximum(0, np.minimum(New_OH_Inventory, Incoming_Order))

    # Put shipments into lefthand shipping slot
    Shipping_flows[0:3, 1] = Outbound_shipments[1:]

    # Send shipments from retailer to the final customer
    Final_Customer_Orders_Filled = Outbound_shipments[0]

    # Update the On-Hand Inventory to account for outflows
    OH_Inventory = New_OH_Inventory - Outbound_shipments

    # Determine Backlog, if any
    Inventory_Shortage = Order_flows[:, 0] - New_OH_Inventory
    New_Backorder = np.maximum(0, Backorder + Inventory_Shortage)
    Backorder = np.copy(New_Backorder)

    # Remember observed order but then Overwrite processed order flow to NaN for debuging if needed
    Observed_Order = np.copy(Order_flows[:, 0])
    #Order_flows[:, 0] = np.nan

    #####Advance Order Slips and Brewers Brew######

    # Advance order slips
    Order_flows[:, 0] = Order_flows[:, (Information_Delay - 1)]
    #Order_flows[:, (Information_Delay - 1)] = np.nan

    # Brewers Brew
    Shipping_flows[3, (Shipping_Delay - 1)] = Production_Request

    #####PLACE ORDERS######

    for i in range(0, 4):

        Entity_Index = i

        # Obsrve the total supply line and the previous demand
        SL = sum(Shipping_flows[Entity_Index, :])
        L = Observed_Order[Entity_Index]

        # L hat is smoothing of observed demand from previous 2 periods
        #if t == 0:
        #    L_hat[Entity_Index] = np.copy(Observed_Order[Entity_Index])

        # Update L_hat (expected future orders) based on observed order
        L_hat_new = theta[Entity_Index] * L + (1 - theta[Entity_Index]) * L_hat[Entity_Index]
        L_hat[Entity_Index] = L_hat_new

        # Note stock of current inventory
        S = OH_Inventory[Entity_Index]

        #Note stock of current inventory inclusive of backorder position
        S = OH_Inventory[Entity_Index] - Backorder[Entity_Index]

        # Add noise to the order if needed
        if (Noisey_Ordering == True):
            eps = np.random.normal(Noise_Mean, Noise_SD)
        else:
            eps = 0

        # AI Decision
        if (AI_Entity == True) and (Entity_Index == AI_Entity_Index):
            if (AI_Order != False):
                #note that AI order is assumed to be free of noise
                Order_Placed = AI_Order
            else:
                Order_Placed = max(0, L_hat[Entity_Index] + alpha_s[Entity_Index] * (
                            S_prime[Entity_Index] - S - beta[Entity_Index] * SL))


        else:
            Order_Placed = max(0, L_hat[Entity_Index] + alpha_s[Entity_Index] * (
                        S_prime[Entity_Index] - S - beta[Entity_Index] * SL) + eps)

        ##TURN ON FOR INTEGER ONLY ORDERING
        if Integer_Ordering == True:
            Order_Placed = round(Order_Placed, 0)

        ##If restricted round, then just place the restricted order
        if Restricted_Ordering == True:
            Order_Placed = np.copy(Restricted_Order)

        if Entity_Index == 3:
            Production_Request = Order_Placed
        else:
            Order_flows[Entity_Index + 1, (Information_Delay - 1)] = np.copy(Order_Placed)

    # End of loop

    # Make orders placed by each entity explict
    Final_Orders[0:3] = Order_flows[1:, (Information_Delay - 1)]
    Final_Orders[3] = Production_Request

    Amplification = (Final_Orders - Orders) / Orders

    # Key variable ot minimize over
    Max_Amp = max(Amplification)
    reward = sum(-25 * np.power(Amplification, 2))

    # fnt_output = list(Order_flows, Shipping_flows, OH_Inventory, Backorder, L_hat, Production_Request, Amplification, reward)
    fnt_output = {"Order_flows": Order_flows, "Shipping_flows": Shipping_flows, "OH_Inventory": OH_Inventory,
                  "Backorder": Backorder, "L_hat": L_hat, "Production_Request": Production_Request,
                  "Entity_Orders": Final_Orders, "Final_Customer_Orders_Filled": Final_Customer_Orders_Filled,
                  "Amplification": Amplification, "reward": reward, "Order_Received": Order_Received}

    return fnt_output


# Function to reset the game to default parameters
def reset_game(horizon=36, Initial_Inventory=12, Information_Delay=2, Shipping_Delay=2):
    # Customer Order String
    # Classic Beer Game
    Step_Round = 4
    Orders = ([4] * Step_Round) + ([9] * (horizon - Step_Round))

    Second_Step = 150
    Orders = Orders[0:Second_Step] + ([9] * (horizon - Second_Step))

    ##################
    # Setting up the game
    ##################

    Order_flows = np.full([4, 2], Orders[0], dtype=float)
    Shipping_flows = np.full([4, 2], Orders[0], dtype=float)
    OH_Inventory = [Initial_Inventory] * 4
    Backorder = [0] * 4
    L_hat = [Orders[0]] * 4
    Order_History = np.full([4, horizon], 0, dtype=float)
    Service_rate = [0] * horizon
    OH_Inventory_History = np.full([4, horizon], 0, dtype=float)
    Backlog_History = np.full([4, horizon], 0, dtype=float)
    Production_Request = Orders[0]

    Amp_Vector = [0] * horizon
    Reward_Vector = [0] * horizon

    Output = {"Orders": Orders, "Order_flows": Order_flows, "Shipping_flows": Shipping_flows,
              "OH_Inventory": OH_Inventory, "Backorder": Backorder, "L_hat": L_hat,
              "Order_History": Order_History, "Service_rate": Service_rate,
              "OH_Inventory_History": OH_Inventory_History, "Backlog_History": Backlog_History,
              "Production_Request": Production_Request, "Amp_Vector": Amp_Vector, "Reward_Vector": Reward_Vector}

    return (Output)


## Main Code

if __name__ == '__main__':

    #Set seed for reproduceability
    Set_Random_Seed = True

    if Set_Random_Seed:
        Random_Seed = 11111111
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(Random_Seed)
        random.seed(Random_Seed)
        tf.random.set_seed(Random_Seed)

    #impot the saved and trained model

    ### Load a saved and trained model

    #model_filename = 'dqn_test_fit.model'
    #model_filename = 'dqn_test_fit_fixed_horizon'
    model_filename = 'dqn_test_fit' #Note that min order here was -20
    model_filename = 'dqn_test_fit_longtrain' #Note that min order here was -20
    #model_filename = 'dqn_test_fit_wide' #Note that min order here was -100

    Relative_Min_Order = -20
    #Relative_Min_Order = -100

    agent = tf.keras.models.load_model(model_filename)

    #Get the implied window size used when originally training the loaded model:
    model_input_shape = (agent.get_layer(index=0).output_shape)[0]  #Get the shape attribute from the input layer
    Original_Batch_Size = model_input_shape[0]                      #First number is the number of items looked as simulateously
    Original_Window_Size = model_input_shape[1]                     #Second number is the window used for any sequential memory
    Original_Observation_Size = model_input_shape[2]                #Third number and (and onwards for multi dimensional inputs) is the actual observed space

    #Main control parameters

    NumRuns = 100
    Noisey_Ordering = True #Dont bother with numruns above if set to false here
    Noise_Mean = 0
    Noise_SD_List = [0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
    Noise_SD_List = [10.5,11,11.5,12,12.5,13,13.5,14,14.5,15]

    Import_Team = True
    Team_Number = 0

    Random_Team = True

    Import_Fitted_Parameters = True
    Fitted_Entity_List = [0,1,2,3] #0 = Retailer, 1 = Wholesaler, 2 = Distributor, 3 = Factory


    horizon = 52
    Holding_Cost = 0.50
    Backorder_Cost = 1.00

    Generate_Baseline = True

    DNN_AI_Entity = False
    DNN_AI_Entity_Index_List = [0,1,2,3]

    Integer_Ordering = True

    # Define if the first few rounds of the game are 'training' with restricted ordering
    Restricted_Order = 4  # Order that all entities are confined to ordering
    Restricted_Rounds = -1  # last round that orders are confined, set to -1 if no restrictions

    if Generate_Baseline == True:
        Runs = ["Baseline","Experiment"]

    else:
        Runs = ["Experiment"]


    #Import the parameters from the '89 Sterman paper
    if Import_Team == True:
        # Import the data from CSV file with ordering decision parameters
        Team_Parameter_Filename = "JS Parameter Table.csv"

        with open(Team_Parameter_Filename, newline='') as csvfile:
            Team_Parameter_Data = list(csv.reader(csvfile))
            All_Team_Parameters = np.asarray(Team_Parameter_Data)

        # Remove header row
        Team_Parameter_Header = All_Team_Parameters[0, :]
        All_Team_Parameters = np.delete(All_Team_Parameters, (0), axis=0)

        # Replace all blanks with 0's
        All_Team_Parameters = np.asarray([[x or '0' for x in xs] for xs in All_Team_Parameters])

        # Extract the team numbers and convert to integers or numbers from strings as appropriate
        Team_Index = [int(item) for item in np.ndarray.tolist(All_Team_Parameters[:, 1])]
        Team_Name = np.ndarray.tolist(All_Team_Parameters[:, 0])
        Entity_Code = np.ndarray.tolist(All_Team_Parameters[:, 2])
        Entity_Index = [int(item) for item in np.ndarray.tolist(All_Team_Parameters[:, 3])]
        thetas = [float(item) for item in np.ndarray.tolist(All_Team_Parameters[:, 4])]
        alphas = [float(item) for item in np.ndarray.tolist(All_Team_Parameters[:, 5])]
        betas = [float(item) for item in np.ndarray.tolist(All_Team_Parameters[:, 6])]
        S_primes = [float(item) for item in np.ndarray.tolist(All_Team_Parameters[:, 7])]

    # Import the parameters fit via a separate box-constraint cost reduced optimization
    if Import_Fitted_Parameters == True:
        Fitted_Parameter_Filename = "Fitted Parameter Table.csv"

        with open(Fitted_Parameter_Filename, newline='') as csvfile:
            Fitted_Parameter_Data = list(csv.reader(csvfile))
            All_Fitted_Parameters = np.asarray(Fitted_Parameter_Data)

        # Remove header row
        Fitted_Parameter_Header = All_Fitted_Parameters[0, :]
        All_Fitted_Parameters = np.delete(All_Fitted_Parameters, (0), axis=0)

        # Replace all blanks with 0's
        All_Fitted_Parameters = np.asarray([[x or '0' for x in xs] for xs in All_Fitted_Parameters])

        # Extract the team numbers and convert to integers or numbers from strings as appropriate
        Fitted_Team_Index = [int(item) for item in np.ndarray.tolist(All_Fitted_Parameters[:, 1])]
        Fitted_Team_Name = np.ndarray.tolist(All_Fitted_Parameters[:, 0])
        Fitted_Entity_Code = np.ndarray.tolist(All_Fitted_Parameters[:, 2])
        Fitted_Entity_Index = [int(item) for item in np.ndarray.tolist(All_Fitted_Parameters[:, 3])]
        Fitted_thetas = [float(item) for item in np.ndarray.tolist(All_Fitted_Parameters[:, 4])]
        Fitted_alphas = [float(item) for item in np.ndarray.tolist(All_Fitted_Parameters[:, 5])]
        Fitted_betas = [float(item) for item in np.ndarray.tolist(All_Fitted_Parameters[:, 6])]
        Fitted_S_primes = [float(item) for item in np.ndarray.tolist(All_Fitted_Parameters[:, 7])]



    #initialize the data capture arrays
    if Import_Fitted_Parameters == True:
        if hasattr(Fitted_Entity_List, "__len__"):
            Historic_Baseline_Costs = np.empty([len(Fitted_Entity_List),NumRuns])
            Historic_Optimzed_Costs = np.empty([len(Fitted_Entity_List),NumRuns])
            Entity_Count = len(Fitted_Entity_List)
        else:
            Historic_Baseline_Costs = np.empty(NumRuns)
            Historic_Optimzed_Costs = np.empty(NumRuns)
            Entity_Count = 1

        # initialize the data capture arrays
    if DNN_AI_Entity == True:
        if hasattr(DNN_AI_Entity_Index_List, "__len__"):
            Historic_Baseline_Costs = np.empty([len(DNN_AI_Entity_Index_List), NumRuns])
            Historic_Optimzed_Costs = np.empty([len(DNN_AI_Entity_Index_List), NumRuns])
            Entity_Count = len(DNN_AI_Entity_Index_List)
        else:
            Historic_Baseline_Costs = np.empty(NumRuns)
            Historic_Optimzed_Costs = np.empty(NumRuns)
            Entity_Count = 1

    CostDiff = np.empty([len(Noise_SD_List),Entity_Count])

    start_time = time.time()

    for noise_index in range(0,len(Noise_SD_List)):

        Noise_SD = Noise_SD_List[noise_index]

        print('Noise value: ' + str(Noise_SD))

        for ent in range(0,Entity_Count):

            print('Entity Index ' + str(ent))


            DNN_AI_Entity_Index = copy.copy(DNN_AI_Entity_Index_List[ent])

            Fitted_Entity = copy.copy(Fitted_Entity_List[ent])

            if (Import_Fitted_Parameters == True) or (DNN_AI_Entity == True):

                if Import_Fitted_Parameters == True:
                    AI_Entity = True
                    AI_Entity_Index = copy.copy(Fitted_Entity)

                if DNN_AI_Entity == True:
                    AI_Entity = True
                    AI_Entity_Index = copy.copy(
                        DNN_AI_Entity_Index)  # note that priority is given to the DNN in the event of a conflict
            else:
                AI_Entity = False

            Backup_AI_Entity = np.copy(AI_Entity)

            # Main loop over the number of runs
            #print('Repeating experiment ' + str(NumRuns) + ' time(s)')
            #if Generate_Baseline == True:
                #print('and repeating for both a baseline and experimental measurement')

            for n in range(0, NumRuns):

                # Get the ordering parameters for other members of the supply chain
                if Import_Team == True:

                    if Random_Team == True:
                        Rand_Team = random.randint(0, max(Team_Index))
                        Team_Number = np.copy(Rand_Team)

                    Team_Mask = np.asarray(Team_Index) == Team_Number

                    Team_Theta = np.asarray(thetas)[Team_Mask]
                    Team_Alpha = np.asarray(alphas)[Team_Mask]
                    Team_Beta = np.asarray(betas)[Team_Mask]
                    Team_S_prime = np.asarray(S_primes)[Team_Mask]

                    # Construct ordering parameter dataframe from the imported parameters
                    Parameter_df = {"theta": np.ndarray.tolist(Team_Theta),
                                    "alpha_s": np.ndarray.tolist(Team_Alpha),
                                    "beta": np.ndarray.tolist(Team_Beta),
                                    "S_prime": np.ndarray.tolist(Team_S_prime)}
                else:
                    # Set default ordering behavior of entities if no external parameters are imported
                    Parameter_df = {"theta": [0.36] * 4,
                                    "alpha_s": [0.26] * 4,
                                    "beta": [0.34] * 4,
                                    "S_prime": [17] * 4}

                # make a back up of the orginal ordering dataframe
                Backup_Parameter_df = Parameter_df.copy()

                # Assemble the ordering parameter dataframe if the fitted entity is included
                if Import_Fitted_Parameters == True:

                    new_thetas = np.asarray(copy.copy(Parameter_df['theta']))
                    new_alphas = np.asarray(copy.copy(Parameter_df['alpha_s']))
                    new_betas = np.asarray(copy.copy(Parameter_df['theta']))
                    new_S_primes = np.asarray(copy.copy(Parameter_df['S_prime']))

                    new_thetas[Fitted_Entity] = copy.copy(Fitted_thetas[Fitted_Entity])
                    new_alphas[Fitted_Entity] = copy.copy(Fitted_alphas[Fitted_Entity])
                    new_betas[Fitted_Entity] = copy.copy(Fitted_betas[Fitted_Entity])
                    new_S_primes[Fitted_Entity] = copy.copy(Fitted_S_primes[Fitted_Entity])

                    # Construct ordering parameter dataframe from the imported parameters
                    Fitted_Parameter_df = {"theta": np.ndarray.tolist(new_thetas),
                                           "alpha_s": np.ndarray.tolist(new_alphas),
                                           "beta": np.ndarray.tolist(new_betas),
                                           "S_prime": np.ndarray.tolist(new_S_primes)}
                else:
                    Fitted_Parameter_df = Backup_Parameter_df.copy()


                for RunName in Runs:

                    if RunName == "Baseline":
                        AI_Entity = False
                        Parameter_df = Backup_Parameter_df.copy()
                    else:
                        AI_Entity = np.copy(Backup_AI_Entity)
                        Parameter_df = Fitted_Parameter_df.copy()

                    AI_Entity = bool(AI_Entity)

                    #Reset the game
                    reset_list = reset_game(horizon=horizon)
                    locals().update(reset_list)

                    #Setup the initial observation for the TensorFlow model
                    #Observed_State = np.array([Agent_Order_Received, Agent_OH_Inventory, Agent_Backorder,
                    #                          Agent_Recent_Order, period, AI_Entity_Index])

                    if DNN_AI_Entity == True:
                        obs = np.array([4, 12, 0,
                                        4, 0, AI_Entity_Index])

                        # Extract the initial steady-state received from the observations set for use in the first instance of relative ordering
                        Agent_Order_Received = obs[1]

                        #Expand initial observation out to fill history or window length
                        obs = np.tile(obs,(Original_Window_Size,1))


                    for t in range(0, (horizon)):

                        if DNN_AI_Entity == True:
                            # Coerce the 1-D or 2-D observation input into a 2-D or 3-D array that TensorFlow will flatten and accept
                            resized_obs = obs[np.newaxis, ...]

                            # Make AI ordering decision based on game state
                            qmatrix = agent.predict(resized_obs)
                            flattened_q = np.ndarray.flatten(qmatrix)
                            BestChoice = np.argmax(flattened_q)

                            Relative_Order = BestChoice + Relative_Min_Order  # + 1 double check this plus one here...
                            Agent_Order = max(0, Agent_Order_Received + Relative_Order)

                            AI_Order = Agent_Order
                        else:
                            AI_Order = False

                        if t <= Restricted_Rounds:
                            Restricted_Ordering = True
                        else:
                            Restricted_Ordering = False

                        # NESTED FUNCTION TO RUN THE GAME FOR ONE TIME STEP AND RETURN THE NEW STATE
                        BeerGame_output = PirateBeerGame_funct(AI_Entity_Index=AI_Entity_Index, AI_Order=AI_Order,
                                                               Orders=Orders[t],
                                                               Order_flows=Order_flows,
                                                               Shipping_flows=Shipping_flows, OH_Inventory=OH_Inventory,
                                                               Backorder=Backorder, L_hat=L_hat,
                                                               Production_Request=Production_Request, AI_Entity=AI_Entity,
                                                               Noisey_Ordering=Noisey_Ordering,
                                                               Noise_Mean=Noise_Mean, Noise_SD=Noise_SD,
                                                               Integer_Ordering=Integer_Ordering,
                                                               Restricted_Ordering = Restricted_Ordering,
                                                               Restricted_Order = Restricted_Order,
                                                               Parameter_df=Parameter_df)

                        locals().update(BeerGame_output)

                        # Write values for analysis/plotting later
                        Order_History[:, t] = np.copy(Entity_Orders)
                        OH_Inventory_History[:, t] = np.copy(OH_Inventory)
                        Backlog_History[:, t] = np.copy(Backorder)
                        Service_rate[t] = np.copy(Final_Customer_Orders_Filled) / np.copy(Orders[t])

                        #Update the observation for the AI
                        #Observed_State = np.array([Agent_Order_Received, Agent_OH_Inventory, Agent_Backorder,
                        #                          Agent_Recent_Order, period, AI_Entity_Index])
                        Agent_Order_Received = np.copy(Order_Received[AI_Entity_Index])
                        Agent_OH_Inventory = np.copy(OH_Inventory[AI_Entity_Index])
                        Agent_Backorder = np.copy(Backorder[AI_Entity_Index])

                        if AI_Entity_Index == 3:
                            Agent_Recent_Order = np.copy(Production_Request)
                        else:
                            Agent_Recent_Order = np.copy(Order_flows[AI_Entity_Index + 1, 1])

                        AI_Entity_Index = AI_Entity_Index

                        if DNN_AI_Entity == True:
                            #AI Observation for this specific time step
                            Observed_State = np.array([Agent_Order_Received, Agent_OH_Inventory, Agent_Backorder,
                                                       Agent_Recent_Order, t, AI_Entity_Index])

                            #Assemble Full AI observation, including hitoric window
                            #Remember, Python indexes from 0 and ignores the final value of a index list!

                            if Original_Window_Size > 1:
                                HistoricObs = np.copy(obs[0:(Original_Window_Size-1)])
                                obs[0] = np.copy(Observed_State)
                                obs[1:] = np.copy(HistoricObs)
                            else:
                                obs = np.copy(Observed_State)

                    if RunName == "Baseline":
                        # Calculate Baseline costs
                        Baseline_Net_Inventory = OH_Inventory_History - Backlog_History
                        Baseline_Costs_Per_Period = OH_Inventory_History * Holding_Cost + Backlog_History * Backorder_Cost
                        Baseline_Total_Costs_Per_Entity = np.sum(Baseline_Costs_Per_Period, 1)
                        Baseline_Total_Team_Costs = sum(Baseline_Total_Costs_Per_Entity)

                        Historic_Baseline_Costs[ent,n] = np.copy(Baseline_Total_Team_Costs)

                    else:
                        # Calculate costs
                        Net_Inventory = OH_Inventory_History - Backlog_History
                        Costs_Per_Period = OH_Inventory_History * Holding_Cost + Backlog_History * Backorder_Cost
                        Total_Costs_Per_Entity = np.sum(Costs_Per_Period, 1)
                        Total_Team_Costs = sum(Total_Costs_Per_Entity)

                        Historic_Optimzed_Costs[ent,n] = np.copy(Total_Team_Costs)

                end_time = time.time()
                ElaspedTime = end_time - start_time
                #print('Run ' + str(n+1) + ' done.')

            print('Finished Entity ' + str(ent) + ' of Noise SD = ' + str(Noise_SD))
            print('Elasped Time: ' + str(round(ElaspedTime, 2)) + ' seconds')


        CostDiff[noise_index,:] = (Historic_Optimzed_Costs.mean(axis=1) - Historic_Baseline_Costs.mean(
            axis=1)) / Historic_Baseline_Costs.mean(axis=1)

        print('Finished SD = ' + str(Noise_SD))
        print('Elasped Time: ' + str(round(ElaspedTime,2)) + ' seconds')



    end_time = time.time()
    print(f"All done in {end_time - start_time}"+ ' seconds')

    #Reporting outputs


    df = pd.DataFrame(CostDiff)
    df.to_excel(excel_writer="C:/Users/jpaine/OneDrive/MIT/Beer Game Project/Python RL/BeerGameDDN/Fixed Stock Management/test.xlsx")