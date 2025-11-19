import cProfile
import numpy as np
from itertools import product
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import CZ, RX, RY, RZ, Identity
from scipy.optimize import minimize
from numpy import load
from hamiltonians.Ham_gen import Hamiltonian_generation
from hamiltonians.JSP import JSP_generation
import copy
import os
import tequila as tq

"""
Environment for executing actions and returning energies and states in the State Representation.
"""


#QuantumStateEnv
class QuantumStateEnv():

    def __init__(self,logger,cfg, main_folder,pred):

        self.cfg=cfg
        self.logger=logger
        Ham_gen=Hamiltonian_generation(cfg,logger,main_folder)
        JSP_Ham=JSP_generation(cfg,logger,main_folder)
        self.molecule=cfg.characteristics["system"]
        if pred:
            print("Hamiltonian generation can take up to a few minutes...")
            self.bond_distance_range = np.arange(cfg.characteristics["start_bond_distance"], cfg.characteristics["end_bond_distance"], 0.01)
        else:
            self.bond_distance_range = np.arange(cfg.characteristics["start_bond_distance"], cfg.characteristics["end_bond_distance"], cfg.characteristics["step_size_bond_distance"]) 
        #path=os.path.join(main_folder, "molecule_data.npy")
        path = os.path.join(main_folder, "molecule_data.npy")
        if self.molecule=="H48" or self.molecule=="H48HF":
            data=Ham_gen.generate_Hamiltonian_H48(self.bond_distance_range)
            np.save(path,np.array(data, dtype=object))
        elif self.molecule=="LiH4" or self.molecule=="LiH6":
            data=Ham_gen.generate_Hamiltonian_LiH(self.bond_distance_range,self.molecule)
            np.save(path,np.array(data, dtype=object))
        elif self.molecule=="JSP":
            data=JSP_Ham.JSP_ham(self.bond_distance_range)
            np.save(path,np.array(data, dtype=object))
        else:
            self.logger.warning("Specified system is unknown. Currently known: H48, LiH4, LiH6, JSP.")

        self.H = [entry[0] for entry in data]  

        self.bond_distances= [entry[2] for entry in data]  

        if len(self.bond_distance_range)!=1:
            self.r_embedding=self.r_to_embedding(self.bond_distances)
    
        self.num_qubits = int(np.log2(np.shape(self.H[0])[0]))
        self.logger.info(f"Qubits: {self.num_qubits}")
        #QUANTUM STATE SIZE: size of the state (networks need to know it as input size), +1 for the layer information
        if len(self.bond_distance_range)!=1:
            self.state_size=2*(2**self.num_qubits)+1+len(self.r_embedding[0])
        else:
            self.state_size=2*(2**self.num_qubits)+1
        #CIRCUIT SIZE (row size +1 entry for the layer) * maximal number of gates
        self.max_gates= self.cfg.training["max_gates"]    
        

        #initalize the qulacs state (needed to apply gates with qulacs)
        self.qustate=QuantumState(self.num_qubits) #QuantumState

        #initialize the qulacs state as numpy-array
        self.state= self.qustate.get_vector()  #QuantumState as nparray
       
        #dictionary of actions
        self.dictionary,self.action_size=self.dictionary()
      


    #DICTIONARY OF ACTIONS
    '''Generates the dictionary of actions, each action is described by an array: [control qubit c, target qubit x, rotation qubit r, rotation axis h]
    rotation axis: 1-> X, 2-> Y, 3-> Z
    --> if control qubit= num_qubits: no CNOT gate
    --> if rotation gate qubit= num_qubits: no Rotation gate
    --> if control qubit=num_qubits and rotation gate qubit= num qubits: Identity
    EXAMPLES (num_qubits=4): [4,0,0,3]: --> apply RZ on qubit 0, [1,2,0,0]: --> apply CNOT on control qubit 1 and target qubit 2
    '''

    #This function generates the dictionary of actions:
    def dictionary(self):
        
        #generate self.dictionary of actions 
        self.dictionary = dict()
        i = 0    


       
        #CNOT actions
        for c in range(self.num_qubits):
            for x in range(self.num_qubits):
                #no CNOT(0,0), CNOT(1,1),..
                if c!=x:
                    self.dictionary[i] =  [c, x, self.num_qubits, 0]
                    i += 1
        
        #Rotation gate actions
        for r, h in product(range(self.num_qubits),
                range(1, 4)):
            self.dictionary[i] = [self.num_qubits, 0, r, h]
            i += 1
        
 
      
        number_of_actions=i-1
     

        return self.dictionary,number_of_actions

    
  

    ###################################################

    #This function applies the chosen action (=gate) on the state:
    def step(self,qustate,chosed_action,angle_action,current_qucircuit,i,layer_scale,index):

        #apply CNOT gate if the first position of the action array does not equal num qubits
        if self.dictionary[chosed_action][0]!=self.num_qubits:
            current_qucircuit.add_CNOT_gate(self.dictionary[chosed_action][0],self.dictionary[chosed_action][1])
            
        #Apply rotation-gate if third position of the action array does not equal num_qubit
        elif self.dictionary[chosed_action][2]!=self.num_qubits:
            angle=[angle_action]#angle initialization
            #Which rotation axis? Build  circuit
            if self.dictionary[chosed_action][3]==1:
                current_qucircuit.add_RX_gate(self.dictionary[chosed_action][2],angle[0])
               
            elif self.dictionary[chosed_action][3]==2:
                current_qucircuit.add_RY_gate(self.dictionary[chosed_action][2],angle[0])
                
            elif self.dictionary[chosed_action][3]==3:
                current_qucircuit.add_RZ_gate(self.dictionary[chosed_action][2],angle[0])
                
            
        #Update the qulacs quantities
        qustate=QuantumState(self.num_qubits)
        current_qucircuit.update_quantum_state(qustate)


    
        #QUANTUM STATE REPRESENTATION
        #get quantum state as numpy array
        state = qustate.get_vector()
        #separate real and imaginary part for the neural net (which can't handle complex numbers)
        nnstate=np.stack([np.real(state),np.imag(state)]).flatten()
        #add the information about the current layer
        outputstate=np.concatenate((nnstate, [layer_scale[i]]))
        #Lastly the information about the bond distance is added, using the embedding function "r_to_embedding"
        if len(self.bond_distance_range)!=1:
            outputstate=np.concatenate((outputstate, self.r_embedding[index]))

        return outputstate,qustate,current_qucircuit
    
       

    #This function calculates the expectation value, i.e energy, currently calculated with the qulacs function for expectation value:
    def get_energy(self,qustate, index):
        state = qustate.get_vector()
        E = np.real(np.vdot(state,np.dot(self.H[index],state)))
        return E
    
  

    
    #define the function that, given the bond distance r, return the value of the embeddings in such point (an array 
    #of $n$ elements, for each $k$)
    def r_to_embedding(self,r):
        r=np.array(r)
        n = self.cfg.gaussian_encoding["number_of_embeddings"]    #number of embeddings
        a = self.cfg.gaussian_encoding["start_interval"]       #left side of the interval
        b = self.cfg.gaussian_encoding["end_interval"]     #right side of the interval
        #crete average of the gaussian for each k
        mu_k = np.linspace(a, b, n)
        #create a standard deviation that is the same for all gaussians
        sigma = (b - a) / n
        return np.exp(-0.5 * ((r[:, np.newaxis] - mu_k) / sigma) ** 2)


    #This function resets the variables for a new episode:
    def reset(self,index):
        current_qucircuit=QuantumCircuit(self.num_qubits)
        if self.cfg.characteristics["hf_start"]=="HF" and self.cfg.characteristics["system"]=="LiH4":
            current_qucircuit.add_X_gate(1)
            current_qucircuit.add_X_gate(0)
        elif self.cfg.characteristics["hf_start"]=="WS" and self.cfg.characteristics["system"]=="LiH4":
            current_qucircuit.add_X_gate(3)
            current_qucircuit.add_X_gate(2)
        elif self.cfg.characteristics["hf_start"]=="HF" and self.cfg.characteristics["system"]=="LiH6":
            current_qucircuit.add_X_gate(0)
            current_qucircuit.add_X_gate(3)
        elif self.cfg.characteristics["hf_start"]=="WS" and self.cfg.characteristics["system"]=="LiH6":
            current_qucircuit.add_X_gate(2)
            current_qucircuit.add_X_gate(5)
        elif self.cfg.characteristics["hf_start"]=="HF" and self.cfg.characteristics["system"]=="H48":
            current_qucircuit.add_X_gate(7)  
            current_qucircuit.add_X_gate(6)  
            current_qucircuit.add_X_gate(3)  
            current_qucircuit.add_X_gate(2)  
        elif self.cfg.characteristics["hf_start"]=="WS" and self.cfg.characteristics["system"]=="H48":
            current_qucircuit.add_X_gate(0)  
            current_qucircuit.add_X_gate(1)  
            current_qucircuit.add_X_gate(5)  
            current_qucircuit.add_X_gate(4)     
        qustate=QuantumState(self.num_qubits)
        current_qucircuit.update_quantum_state(qustate)
        state = qustate.get_vector()
        current_circuit=[]
        angles=[]
        #Initialize Quantum State representation
        nnstate=np.stack([np.real(state),np.imag(state)]).flatten()
        outputstate=np.concatenate((nnstate, [-1]))
        if len(self.bond_distance_range)!=1:
            outputstate=np.concatenate((outputstate, self.r_embedding[index]))
     
        return outputstate,qustate,current_qucircuit,current_circuit,angles
