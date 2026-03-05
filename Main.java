/*
* Author: Tarush Gupta
* Date: February 27, 2026
* Description: Main driver class for the ABC Neural Net.
*              Execution flow driven entirely by setConfig().
*              Timing is measured outside train/run and stored for reporting.
*/
public class Main
{
/*
* Runs. 
*/
   public static void main(String[] args)
   {
      ABC network = new ABC();

      network.setConfig();

      network.echoConfig();

      network.allocateMemory();

      network.populateArrays();

/*
* Train if the network is training.
*/
      if (network.isTraining)
      {
         long trainStart = System.currentTimeMillis();
         network.train();
         network.trainTimeMs = System.currentTimeMillis() - trainStart;
         network.reportTrainingResults();
      }
      else
      {
         long runStart = System.currentTimeMillis();
         network.runTestCases();
         network.trainTimeMs = System.currentTimeMillis() - runStart;
         network.reportTrainingResults();
      }

/*
* Run if the network is running.
*/
      if (network.shouldRun)
      {
         long runStart = System.currentTimeMillis();
         if (network.isTraining)
         {
            network.runTrainingCases();
         }
         else
         {
            network.runTestCases();
         }
         network.runTimeMs = System.currentTimeMillis() - runStart;
         network.reportRunResults();
      } // if (network.shouldRun)

/*
* Save weights to binary file if configured.
*/
      if (network.saveFile != null && network.saveFile.length() > 0)
      {
         network.saveWeights();
      }
   } // main(String[] args)
} // class Main
