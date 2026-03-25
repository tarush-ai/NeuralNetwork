/*
* Author: Tarush Gupta
* Date: March 12, 2026
* Description: Main driver class for the ABCbp Neural Net.
*              Execution flow driven entirely by config.json at runtime.
*              Timing is measured outside train/run and stored for reporting.
*/
public class Main
{
/*
* Entry point for the ABCbp neural network. Reads config from file, allocates,
* populates, trains or runs the network, and optionally saves weights.
*/
   public static void main(String[] args) throws Exception
   {
      String configFile = "config.json";

      ABCbp network = new ABCbp();

      network.setConfig(configFile);

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
      } // if (network.isTraining)

/*
* Run if the network is running.
*/
      if (network.shouldRun)
      {
         long runStart = System.currentTimeMillis();
         network.runTestCases();
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
