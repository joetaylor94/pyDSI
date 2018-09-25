
#include "IpIpoptApplication.hpp"
#include "condmodel_nlp.hpp"


using namespace Ipopt;


int main(int argv, char* argc[])
{
  // Create a new instance of your nlp 
  // (use a SmartPtr, not raw)
  SmartPtr<TNLP> mynlp = new CONDMODEL_NLP();
    

  // Create a new instance of IpoptApplication
  //  (use a SmartPtr, not raw)
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", 1e-9);
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("output_file", "ipopt.out");
    

  // Intialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded) {
    printf("*** Error during initialization!");
    return (int) status;
  }
    
  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(mynlp);
    
  if (status == Solve_Succeeded) {
    printf("*** The problem solved!");
  }
  else {
    printf("*** The problem FAILED!");
  }
    
  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically 
  // be deleted.
    

  return(int) status;
}