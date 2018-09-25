        // Test_main.cpp
        // IPOPT code for variational inference
        // Author:  Joseph Taylor
        
        #include <cstdio>
        #include <stdio.h>    
        #include <stdlib.h>
        #include <time.h>  
        
        #include "IpIpoptApplication.hpp"
        #include "Test_nlp.hpp" 
        
        using namespace Ipopt;
        
        int main(int argv, char* argc[])
        {
          
          // Create a new instance of your nlp
          //  (use a SmartPtr, not raw)
          SmartPtr<TNLP> mynlp = new Test_NLP(); 
        
          // Create a new instance of IpoptApplication
          //  (use a SmartPtr, not raw)
          SmartPtr<IpoptApplication> app = new IpoptApplication();
        
          // Change some options
          // Note: The following choices are only examples, they might not be
          //       suitable for your optimization problem.
          app->Options()->SetNumericValue("tol", 1e-10);
          app->Options()->SetStringValue("mu_strategy", "adaptive");
          app->Options()->SetStringValue("output_file", "../output/ipopt.out");
        
          // Intialize the IpoptApplication and process the options
          app->Initialize();
        
          // Ask Ipopt to solve the problem
          ApplicationReturnStatus status = app->OptimizeTNLP(mynlp);
        
          if (status == Solve_Succeeded) {
            printf("\n\n*** The problem solved!\n");
          }
          else {
            printf("\n\n*** The problem FAILED!\n");
          }
        
          return (int) status;
        }
        
        
