import pdb
import os
import wntr
import copy
import ray
from wntr.network.model import *
from wntr.sim.solvers import *
import matplotlib.pyplot as plt
from  scipy.optimize import minimize

class EnsembleKalmanInversion():
    def __init__(self, wn, head_obs, sensor_location, corr_demands, corr_reservoir_nodes, base_demand, std_dev):
        self.wn = wn
        self.head_obs = head_obs
        self.sensor_location = sensor_location
        self.corr_demands = corr_demands
        self.corr_reservoir_nodes = corr_reservoir_nodes
        self.base_demand = base_demand
        self.std_dev = std_dev

    def forward_model(self, pars):

        demand = pars['demand']
        roughness = pars['roughness']

        flowrate, head = run_simulation(self.wn, demand, roughness)

        results = head[self.sensor_location]

        return results

    def compute_empircal_par_mean(self, pars):
        emp_mean = np.mean(pars, axis=0)
        return emp_mean
    
    def compute_empiral_odel_output_mean(self, pars, emp_mean):
        emp_cov = np.cov(pars, rowvar=False)
        return emp_cov
    
    def compute_CPP(self, pars, emp_mean, emp_cov):
        CPP = np.zeros((len(pars), len(pars[0])))
        for i in range(len(pars)):
            CPP[i] = np.dot(np.dot((pars[i] - emp_mean), np.linalg.inv(emp_cov)), (pars[i] - emp_mean).T)
        return CPP

# function to create a covariance matrix
def cov_mat_fixed(corr_demands, corr_reservoir_nodes):
    N = num_nodes

    mat_corr = np.zeros((N, N))  # initializing matrix

    mat = np.full((N - 1, N - 1),
                  corr_demands)  # step 1 for a symmetric matrix of n-1 by n-1
    mat_symm = (mat + mat.T) / 2  # step 2

    diag = np.ones(N)  # setting up the diagonal matrix, variance of nodal demands
    np.fill_diagonal(mat_symm, diag)
    mat_corr[1:, 1:] = mat_symm

    mat_corr[0, 0] = 1  # element (0,0) which is variance of resevoir head

    top = np.full((N - 1),corr_reservoir_nodes)  # covariance between reservoir head and nodal demands
   
    mat_corr[0, 1:] = top
    mat_corr[1:, 0] = top

    Diag = np.diag(std_dev)
    cov_mat = Diag * mat_corr * Diag

    return cov_mat


def run_simulation(roughness, demand, wn):

    for link in wn.links:
        wn.links[link].roughness = roughness

    #getting samples
    #train_data_raw = np.random.multivariate_normal(base_demand,cov_mat,1)
    train_data_raw = demand

    #removing samples with negative values
    #train_data_raw_positive = train_data_raw[train_data_raw.min(axis=1)>=0,:]
    train_data_raw_positive = train_data_raw[train_data_raw.min(axis=1)>=0,:]

    #creating numpy arrays to store EPANET simulation output
    train_samples_positive = train_data_raw_positive.shape[0]

    # updating reservoir head in the epanet input
    wn.get_node(1).head_timeseries.base_value = train_data_raw_positive[0,0]

    # updating nodal demand for all nodes in the epanet input
    j=1
    for n in wn.nodes.junction_names:
        wn.get_node(n).demand_timeseries_list[0].base_value = train_data_raw_positive[0,j]
        j=j+1

    #lol = HydraulicModel(wn)
    sim = wntr.sim.WNTRSimulator(wn)
    # storing simulation results in 3 matrices
    results = sim.run_sim()

    flowrate = results.link['flowrate']
    head = results.node['head']

    return flowrate, head

def compute_sensor_residual(
    roughness,
    cov_mat, 
    base_demand, 
    wn,
    head_obs,
    sensor_location,
):
    flowrate, head = run_simulation(
        roughness=roughness,
        cov_mat=cov_mat, 
        base_demand=base_demand,
        wn=wn
        )

    pred_head_obs = head[sensor_location]
    pred_head_obs = pred_head_obs.values

    head_obs = head_obs.values

    residual = np.linalg.norm(pred_head_obs - head_obs)

    #np.sum(np.log(2*math.pi*(sigma**2))/2 + ((data-mu)**2)/(2 * (sigma**2)))
    return residual

def objective_function(
    roughness,
    corr_demands,
    corr_reservoir_nodes,        
    base_demand, 
    wn,
    head_obs,
    sensor_location,
):
    covmat_base = cov_mat_fixed(corr_demands, corr_reservoir_nodes)

    res = []
    for i in range(head_obs.shape[0]):
        residual = compute_sensor_residual(
            roughness=roughness,
            cov_mat=covmat_base, 
            base_demand=base_demand, 
            wn=wn,
            head_obs=head_obs.iloc()[i],
            sensor_location=sensor_location,
        )
        res.append(residual)
    return np.mean(res)


if __name__ == "__main__":
    # Getting path for the input file
    inputfiles_folder_name = 'Input_files_EPANET'
    filename = 'Hanoi_base_demand.inp'
    path_file = os.path.join(inputfiles_folder_name,filename)

    # Load the wntr model
    inp_file = path_file
    wn = wntr.network.WaterNetworkModel(inp_file)

    # store no of nodes and links
    num_nodes = len(wn.node_name_list)
    num_links = len(wn.link_name_list)

    # create array that contains the base reservoir head and demand at nodes
    base_demands = np.zeros((num_nodes))
    base_demands[0]= wn.get_node(1).head_timeseries.base_value
    for i in range(1,num_nodes):
        base_demands[i] = wn.get_node(i+1).demand_timeseries_list[0].base_value

    # define standard deviation matrix
    std_dev = base_demands*0.2
    std_dev[0] = base_demands[0]*0.05

    # Covmat for all experiments
    covmat_base = cov_mat_fixed(0.6, 0.5)

    head_obs = []
    for i in range(50):
        demand = np.random.multivariate_normal(base_demands, covmat_base, 1)
        flowrate, head = run_simulation(
            roughness=130,
            demand=demand,
            wn=wn
            )
        sensor_location = list(head.columns[0:-1:5])
        head_obs.append(head[sensor_location])
    head_obs = pd.concat(head_obs)

    objective = lambda x: objective_function(
        roughness=x[0],
        corr_demands=0.6,
        corr_reservoir_nodes=0.5,
        base_demand=base_demands,
        wn=wn,
        head_obs=head_obs,
        sensor_location=sensor_location,
    )
        
    x_init = np.array([100])#.4, .4])
    
    x_min = minimize(
        objective,
        x_init,
        method="L-BFGS-B",
        bounds=[(80, 150)],
    )
    pdb.set_trace()
        
    
