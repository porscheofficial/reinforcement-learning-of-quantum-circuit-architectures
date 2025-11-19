from qiskit.quantum_info import SparsePauliOp
import numpy as np
from numpy import linalg as LA

class JSP_generation():

    def __init__(self,cfg,logger,main_folder):
        self.cfg=cfg
        self.logger=logger
        self.main_folder=main_folder

    def JSP_ham(self,L3):

        JSP_info=[]

        for q in range(len(L3)):
            L=[1,1,L3[q]] #length of jobs
            N=3 #jobs
            m=2 #machines
            M=1 #max runtime difference
            A=4 #prefactor constrains
            B=1 #prefactor optimization
            num_qubits=N*m+(m-1)*M

            identity = SparsePauliOp.from_list([('I' * num_qubits,1)]).to_matrix()
            operator_list=self.binary_variable_operators(num_qubits)

            #first term
            first_term=np.zeros((2**num_qubits,2**num_qubits))
            help_term_2=np.zeros((2**num_qubits,2**num_qubits))
            for i in range(N):
                help_term_1=np.zeros((2**num_qubits,2**num_qubits))
                for j in range(m):
                    help_term_1=help_term_1+operator_list[i+j*N].to_matrix()
                first_term=first_term+(identity-help_term_1)**2

            #second term
            second_term=np.zeros((2**num_qubits,2**num_qubits))
            for j in range(1,m):
                help_term_3=np.zeros((2**num_qubits,2**num_qubits))
                for i in range(N):
                    help_term_3=help_term_3+L[i]*(operator_list[i+j*N].to_matrix()-operator_list[i].to_matrix())

                help_term_4=np.zeros((2**num_qubits,2**num_qubits))
            
                for p in range(0,M):
                    help_term_4=help_term_4+(p+1)*operator_list[m*N+p+(j-1)*M].to_matrix()
                    
                second_term=second_term+(help_term_3+help_term_4)**2

            #Opt term
            opt_term=np.zeros((2**num_qubits,2**num_qubits))
            for i in range(N):
                opt_term=opt_term+L[i]*operator_list[i].to_matrix()

            H=A*first_term+A*second_term+B*opt_term

            v,vv=LA.eigh(H)

            JSP_info.append((H,v[0],np.round(L3[q],2)))

        return JSP_info

    def binary_variable_operators(self,num_qubits):

        identity = SparsePauliOp.from_list([('I' * num_qubits, 0.5)])
    
        operator_list=[]
        for i in range(num_qubits):

            pauli_str = ['I'] * num_qubits
            pauli_str[i] = 'Z'
            label = ''.join(pauli_str)
            z_term = SparsePauliOp.from_list([(label, -0.5)])
            
            operator_list.append(identity + z_term)
        return operator_list

