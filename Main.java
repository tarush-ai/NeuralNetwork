/*
* Author: Tarush Gupta
* Date: April 29, 2026
* Description: Main driver class for the N-layer neural network.
*              Reads the configuration file name from the command line, or uses
*              the default if none is supplied. Allocates, populates, trains or
*              runs the network, and optionally saves weights.
*/
public class Main
{
/*
* Default configuration file used when no command-line argument is supplied.
*/
   static final String DEFAULT_CONFIG_FILE = "config.json";

/*
* Entry point for the N-layer neural network. Reads config from file, allocates,
* populates, trains or runs the network, and optionally saves weights.
*/
   public static void main(String[] args) throws Exception
   {
      String configFile;

      if (args.length == 0 || args[0].length() == 0)
      {
         configFile = DEFAULT_CONFIG_FILE;
      }
      else
      {
         configFile = args[0];
      }

      NLayer network = new NLayer();

      network.setConfig(configFile);
      network.echoConfig();
      network.allocateMemory();
      network.populateArrays();

/*
* Train if the network is in training mode.
*/
      if (network.isTraining)
      {
         long trainStart = System.currentTimeMillis();
         network.train();
         network.trainTimeMs = System.currentTimeMillis() - trainStart;
         network.reportTrainingResults();
      } // if (network.isTraining)

/*
* Run if the network is in running mode.
*/
      if (network.shouldRun)
      {
         long runStart = System.currentTimeMillis();
         network.runTestCases();
         network.runTimeMs = System.currentTimeMillis() - runStart;
         network.reportRunResults();
      } // if (network.shouldRun)

/*
* Save weights to binary file if a save path is configured.
*/
      if (network.saveFile != null && network.saveFile.length() > 0)
      {
         network.saveWeights();
      }
   } // main(String[] args)
} // class Main
