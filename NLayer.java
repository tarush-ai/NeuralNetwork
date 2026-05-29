/*
* Author: Tarush Gupta
* Date: April 29, 2026
* Description: N-activation-layer neural network with backpropagation.
*              Generalizes to any number of layers with any number of nodes per layer.
*              Layer sizes, training parameters, and file paths are all selected at
*              runtime via a JSON configuration file with a comma-separated "layers"
*              field (e.g., "2,20,3"). Supports random weight initialization or binary
*              weight load/save, and training or running modes.
*
* Table of Contents:
* public double activation(double x)
* public double activationDerivative(double x)
* public void setConfig(String configFile)
* private String getJsonValue(String json, String key)
* public void echoConfig()
* public void allocateMemory()
* private void loadInputs(String filename)
* private void loadTruthTable(String filename)
* public void populateArrays()
* public void printInputTable()
* public void printTruthTable()
* public void randomizeWeights()
* public void useLoadedWeights()
* public void saveWeights()
* public double randomize(double low, double high)
* public double sigmoid(double x)
* public double sigmoidDerivative(double x)
* public double tanh(double x)
* public double tanhDerivative(double x)
* public double linear(double x)
* public double linearDerivative(double x)
* public double runTrainCase(int caseIndex)
* public void runTestCase()
* public void train()
* public void reportTrainingResults()
* public void populateInput(int caseIndex)
* public void runTestCases()
* public void reportRunResults()
*/
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;

public class NLayer
{
   static final int INPUT_LAYER   = 0;
   static final int HIDDEN_FIRST  = 1;
   static final int HIDDEN_SECOND = 2;

/*
* Network configuration
*/
   public int numLayers;
   public int[] layerSizes;

/*
* Training hyperparameters.
*/
   public double lambda;
   public double low;
   public double high;
   public int maxIterations;
   public double errorThreshold;
   public boolean useRandomWeights;
   public int numCases;
   public int keepAlive;

/*
* Network arrays.
*/
   public double[][]  a;
   public double[][][] w;
   public double[][] inputs;

/*
* Run results
*/
   public double[][] results;

/*
* Training-only arrays.
*/
   public double[][] targets;
   public double[][] theta;
   public double[][] psi;

/*
* Training results
*/
   public int  trainIterations;
   public double trainError;
   public long trainTimeMs;
   public long runTimeMs;

/*
* Config file and runtime configuration
*/
   public boolean isTraining;
   public boolean shouldRun;
   public boolean showInputTable;
   public boolean showTruthTable;
   public String loadFile;
   public String saveFile;
   public String inputFile;
   public String truthTableFile;
   public String configFile;

/*
* Threshold function. Change this line (and activationDerivative below)
* to switch activation functions: sigmoid(x), tanh(x), or linear(x).
*/
   public double activation(double x)
   {
      return sigmoid(x);
   } // activation(double x)

/*
* Threshold function derivative. Change this line to match the choice in
* activation above: sigmoidDerivative(x), tanhDerivative(x), or linearDerivative(x).
*/
   public double activationDerivative(double x)
   {
      return sigmoidDerivative(x);
   } // activationDerivative(double x)

/*
* End of the user configuration.
*/

/*
* Reads all network parameters from the given JSON config file.
* The "layers" key must be a comma-separated string of layer sizes (e.g., "2,20,3").
*/
   public void setConfig(String configFile) throws Exception
   {
      this.configFile = configFile;

      StringBuilder sb = new StringBuilder();
      BufferedReader reader = new BufferedReader(new FileReader(configFile));
      String line;
      while ((line = reader.readLine()) != null)
      {
         sb.append(line);
      }
      reader.close();
      String json = sb.toString();

      String layersStr = getJsonValue(json, "layers");
      String[] parts = layersStr.split(",");
      numLayers = parts.length;
      layerSizes = new int[numLayers];

      for (int i = 0; i < numLayers; ++i)
      {
         layerSizes[i] = Integer.parseInt(parts[i].trim());
      }

      numCases = Integer.parseInt(getJsonValue(json, "numCases"));
      inputFile = getJsonValue(json, "inputFile");
      truthTableFile = getJsonValue(json, "truthTableFile");
      isTraining = Boolean.parseBoolean(getJsonValue(json, "isTraining"));
      useRandomWeights = Boolean.parseBoolean(getJsonValue(json, "useRandomWeights"));
      low = Double.parseDouble(getJsonValue(json, "low"));
      high = Double.parseDouble(getJsonValue(json, "high"));
      lambda = Double.parseDouble(getJsonValue(json, "lambda"));
      maxIterations = Integer.parseInt(getJsonValue(json, "maxIterations"));
      errorThreshold = Double.parseDouble(getJsonValue(json, "errorThreshold"));
      keepAlive = Integer.parseInt(getJsonValue(json, "keepAlive"));
      shouldRun = Boolean.parseBoolean(getJsonValue(json, "shouldRun"));
      showInputTable = Boolean.parseBoolean(getJsonValue(json, "showInputTable"));
      showTruthTable = Boolean.parseBoolean(getJsonValue(json, "showTruthTable"));
      loadFile = getJsonValue(json, "loadFile");
      saveFile = getJsonValue(json, "saveFile");
   } // setConfig(String configFile)

/*
* Extracts the raw value string for the given key from a JSON string.
* Matches "key" only when followed by ':' so embedded copies inside string
* values are skipped. Returns empty string if the key is not found.
*/
   private String getJsonValue(String json, String key)
   {
      String result = "";
      String pattern = "\"" + key + "\"";
      int patternLen = pattern.length();
      int len = json.length();
      int keyIdx = -1;
      int searchFrom = 0;

      while (searchFrom != -1 && keyIdx == -1)
      {
         int idx = json.indexOf(pattern, searchFrom);

         if (idx == -1)
         {
            searchFrom = -1;
         }
         else
         {
            int p = idx + patternLen;
            while (p < len && Character.isWhitespace(json.charAt(p)))
            {
               ++p;
            }

            if (p < len && json.charAt(p) == ':')
            {
               keyIdx = idx;
            }
            else
            {
               searchFrom = idx + 1;
            }
         }
      } // while (searchFrom != -1 && keyIdx == -1)

      if (keyIdx != -1)
      {
         int colonIdx = json.indexOf(":", keyIdx);
         String rest = json.substring(colonIdx + 1).trim();

         if (rest.charAt(0) == '"')
         {
            int openQuote = json.indexOf('"', colonIdx + 1);
            int closeQuote = json.indexOf('"', openQuote + 1);
            result = json.substring(openQuote + 1, closeQuote);
         }
         else
         {
            int commaIdx = json.indexOf(",", colonIdx);
            int braceIdx = json.indexOf("}", colonIdx);
            int endIdx = (commaIdx != -1 && commaIdx < braceIdx) ? commaIdx : braceIdx;
            result = json.substring(colonIdx + 1, endIdx).trim();
         }
      } // if (keyIdx != -1)

      return result;
   } // getJsonValue(String json, String key)

/*
* Prints current network config including layer sizes, file names, and training parameters when applicable.
*/
   public void echoConfig()
   {
      int n;

      System.out.print("Config File: " + configFile + "\nNetwork Configuration: ");
      for (n = 0; n < numLayers; ++n)
      {
         if (n > 0) System.out.print("-");
         System.out.print(layerSizes[n]);
      }
      System.out.println();
      System.out.println("Number of Test Cases: " + numCases);
      System.out.println("Input File: " + inputFile);
      System.out.println("Mode: " + (isTraining ? "Training" : "Running"));
      System.out.println("Running: " + shouldRun);
      System.out.println("Keep Alive: " + keepAlive);

      if (isTraining)
      {
         System.out.println("\nTraining Parameters:");
         System.out.println("Truth Table File: " + truthTableFile);
         System.out.println("Weight Initialization: " + (useRandomWeights ? "Random" : "Loaded"));

         if (useRandomWeights)
         {
            System.out.println("Random Number Range: [" + low + ", " + high + "]");
         }

         System.out.println("Max Iterations: " + maxIterations);
         System.out.println("Error Threshold: " + errorThreshold);
         System.out.println("Lambda: " + lambda);
      } // if (isTraining)

      if (loadFile != null && loadFile.length() > 0)
      {
         System.out.println("Load File: " + loadFile);
      }

      if (saveFile != null && saveFile.length() > 0)
      {
         System.out.println("Save File: " + saveFile);
      }
   } // echoConfig()

/*
* Allocates memory for all network arrays. Training-only arrays (theta, psi, targets)
* are only allocated when the network is in training mode.
*/
   public void allocateMemory()
   {
      int n;
      int outputLayer = numLayers - 1;
      int hiddenLast = numLayers - 2;

      a = new double[numLayers][];
      for (n = 0; n < numLayers; ++n)
      {
         a[n] = new double[layerSizes[n]];
      }

      w = new double[numLayers - 1][][];
      for (n = 0; n < numLayers - 1; ++n)
      {
         w[n] = new double[layerSizes[n]][layerSizes[n + 1]];
      }

      inputs = new double[numCases][layerSizes[INPUT_LAYER]];

      if (!isTraining || shouldRun)
      {
         results = new double[numCases][layerSizes[outputLayer]];
      }

      if (isTraining)
      {
         targets = new double[numCases][layerSizes[outputLayer]];

         theta = new double[numLayers][];
         for (n = HIDDEN_FIRST; n <= hiddenLast; ++n)
         {
            theta[n] = new double[layerSizes[n]];
         }

         psi = new double[numLayers][];
         for (n = HIDDEN_SECOND; n < numLayers; ++n)
         {
            psi[n] = new double[layerSizes[n]];
         }
      } // if (isTraining)
   } // allocateMemory()

/*
* Reads binary input activations from the given file into the inputs array.
*/
   private void loadInputs(String filename) throws Exception
   {
      DataInputStream in = new DataInputStream(new FileInputStream(filename));

      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < layerSizes[INPUT_LAYER]; ++m)
         {
            inputs[c][m] = in.readDouble();
         }
      }

      in.close();
   } // loadInputs(String filename)

/*
* Reads a truth table from the given file into the targets array.
*/
   private void loadTruthTable(String filename) throws Exception
   {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      int outputLayer = numLayers - 1;

      for (int c = 0; c < numCases; ++c)
      {
         String[] tokens = br.readLine().trim().split("\\s+");
         for (int i = 0; i < layerSizes[outputLayer]; ++i)
         {
            targets[c][i] = Double.parseDouble(tokens[i]);
         }
      }

      br.close();
   } // loadTruthTable(String filename)

/*
* Loads inputs and (if training) targets, then initializes weights from
* random values or a saved binary file as configured.
*/
   public void populateArrays() throws Exception
   {
      loadInputs(inputFile);

      if (isTraining)
      {
         loadTruthTable(truthTableFile);
      }

      if (useRandomWeights)
      {
         randomizeWeights();
      }
      else if (loadFile != null && loadFile.length() > 0)
      {
         useLoadedWeights();
      }
   } // populateArrays()

/*
* Prints the input table by iterating through the inputs.
*/
   public void printInputTable()
   {
      System.out.println("\nInput Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < layerSizes[INPUT_LAYER]; ++m)
         {
            System.out.print(inputs[c][m] + "\t");
         }
         System.out.println();
      }
   } // printInputTable()

/*
* Prints the truth table by iterating through the inputs and targets,
* showing all output columns for each test case.
*/
   public void printTruthTable()
   {
      int outputLayer = numLayers - 1;

      System.out.println("\nTruth Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < layerSizes[INPUT_LAYER]; ++m)
         {
            System.out.print(inputs[c][m] + "\t");
         }
         System.out.print("| ");
         for (int i = 0; i < layerSizes[outputLayer]; ++i)
         {
            System.out.print(targets[c][i] + "\t");
         }
         System.out.println();
      } // for (int c = 0; c < numCases; ++c)
   } // printTruthTable()

/*
* Initializes all weights randomly using the randomize function.
*/
   public void randomizeWeights()
   {
      int n;
      int j;
      int k;

      for (n = 0; n < numLayers - 1; ++n)
      {
         for (j = 0; j < layerSizes[n]; ++j)
         {
            for (k = 0; k < layerSizes[n + 1]; ++k)
            {
               w[n][j][k] = randomize(low, high);
            }
         }
      } // for (n = 0; n < numLayers - 1; ++n)
   } // randomizeWeights()

/*
* Uses the weights from the loaded file.
*/
   public void useLoadedWeights()
   {
      try
      {
         DataInputStream in = new DataInputStream(new FileInputStream(loadFile));
         int fileNumLayers = in.readInt();
         int[] fileSizes = new int[fileNumLayers];
         boolean match;
         int n;
         int j;
         int k;

         for (n = 0; n < fileNumLayers; ++n)
         {
            fileSizes[n] = in.readInt();
         }

         match = (fileNumLayers == numLayers);
         for (n = 0; n < numLayers; ++n)
         {
            if (fileSizes[n] != layerSizes[n]) match = false;
         }

         if (match)
         {
            for (n = 0; n < numLayers - 1; ++n)
            {
               for (j = 0; j < layerSizes[n]; ++j)
               {
                  for (k = 0; k < layerSizes[n + 1]; ++k)
                  {
                     w[n][j][k] = in.readDouble();
                  }
               }
            } // for (n = 0; n < numLayers - 1; ++n)
         } // if (match)
         else
         {
            System.out.println("Error: weight file config does not match network config.");
         } // else

         in.close();
      } // try
      catch (Exception e)
      {
         System.out.println("Error loading weights: " + e.getMessage());
      }
   } // useLoadedWeights()

/*
* Saves the weights to the relevant file.
*/
   public void saveWeights()
   {
      try
      {
         DataOutputStream out = new DataOutputStream(new FileOutputStream(saveFile));
         int n;
         int j;
         int k;

         out.writeInt(numLayers);
         for (n = 0; n < numLayers; ++n)
         {
            out.writeInt(layerSizes[n]);
         }

         for (n = 0; n < numLayers - 1; ++n)
         {
            for (j = 0; j < layerSizes[n]; ++j)
            {
               for (k = 0; k < layerSizes[n + 1]; ++k)
               {
                  out.writeDouble(w[n][j][k]);
               }
            }
         } // for (n = 0; n < numLayers - 1; ++n)

         out.close();
      } // try
      catch (Exception e)
      {
         System.out.println("Unable to save weights to " + saveFile);
      }
   } // saveWeights()

/*
* Returns a random double between low and high. It uses
* Math.random().
*/
   public double randomize(double low, double high)
   {
      return low + Math.random() * (high - low);
   } // randomize(double low, double high)

/*
* Sigmoid activation: f(x) = 1 / (1 + e^-x). Uses Math.exp().
*/
   public double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // sigmoid(double x)

/*
* Sigmoid activation derivative: f'(x) = f(x) * (1 - f(x)).
*/
   public double sigmoidDerivative(double x)
   {
      double val = activation(x);
      return val * (1.0 - val);
   } // sigmoidDerivative(double x)

/*
* Hyperbolic tangent activation using eps(x) = (x < 0) ? 1 : -1. exp(eps * 2x) is computed
* once and reused in both the numerator and denominator.
*/
   public double tanh(double x)
   {
      double eps = (x < 0) ? 1.0 : -1.0;
      double e   = Math.exp(eps * 2.0 * x);
      return eps * (e - 1.0) / (e + 1.0);
   } // tanh(double x)

/*
* Hyperbolic tangent derivative: f'(x) = 1 - f^2(x).
*/
   public double tanhDerivative(double x)
   {
      double val = tanh(x);
      return 1.0 - val * val;
   } // tanhDerivative(double x)

/*
* Linear activation: f(x) = x
*/
   public double linear(double x)
   {
      return x;
   } // linear(double x)

/*
* Linear activation derivative: f'(x) = 1.
*/
   public double linearDerivative(double x)
   {
      return 1.0;
   } // linearDerivative(double x)

/*
* Forward pass for training. Returns twice the total squared error for the given case.
*/
   public double runTrainCase(int caseIndex)
   {
      int n;
      int j;
      int k;
      int outputLayer = numLayers - 1;
      int hiddenLast  = numLayers - 2;
      double twiceError  = 0.0;

/*
* Hidden layers: theta[n][j] = sum_k a[n-1][k] * w[n-1][k][j], a[n][j] = f(theta[n][j]).
*/
      for (n = HIDDEN_FIRST; n <= hiddenLast; ++n)
      {
         for (j = 0; j < layerSizes[n]; ++j)
         {
            theta[n][j] = 0.0;
            for (k = 0; k < layerSizes[n - 1]; ++k)
            {
               theta[n][j] += a[n - 1][k] * w[n - 1][k][j];
            }
            a[n][j] = activation(theta[n][j]);
         }
      } // for (n = HIDDEN_FIRST; n <= hiddenLast; ++n)

/*
* Output layer: omega = T - a, psi = omega * f'(theta), 2E += omega^2.
*/
      n = outputLayer;
      for (j = 0; j < layerSizes[n]; ++j)
      {
         double thetaJ = 0.0;
         for (k = 0; k < layerSizes[n - 1]; ++k)
         {
            thetaJ += a[n - 1][k] * w[n - 1][k][j];
         }
         a[n][j] = activation(thetaJ);
         double omegaJ = targets[caseIndex][j] - a[n][j];
         psi[n][j] = omegaJ * activationDerivative(thetaJ);
         twiceError += omegaJ * omegaJ;
      } // for (j = 0; j < layerSizes[n]; ++j)

      return twiceError;
   } // runTrainCase(int caseIndex)

/*
* Runs the network forward on pre-populated input activations.
*/
   public void runTestCase()
   {
      int n;
      int j;
      int k;

      for (n = HIDDEN_FIRST; n < numLayers; ++n)
      {
         for (j = 0; j < layerSizes[n]; ++j)
         {
            double thetaAcc = 0.0;
            for (k = 0; k < layerSizes[n - 1]; ++k)
            {
               thetaAcc += a[n - 1][k] * w[n - 1][k][j];
            }
            a[n][j] = activation(thetaAcc);
         }
      } // for (n = HIDDEN_FIRST; n < numLayers; ++n)
   } // runTestCase()

/*
* Trains the network using steepest descent across all training cases.
* Stores final iteration count and error in trainIterations and trainError.
*/
   public void train()
   {
      int iteration = 0;
      double avgError  = Double.MAX_VALUE;
      int caseIndex;
      int n;
      int j;
      int k;
      int m;
      int hiddenLast = numLayers - 2;
      double totalError;

      while (iteration < maxIterations && avgError >= errorThreshold)
      {
         totalError = 0.0;

         for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
         {
            populateInput(caseIndex);
            totalError += runTrainCase(caseIndex);

/*
* HiddenLast..HiddenSecond: Omega = sum_k psi[n+1][k] * w[n][j][k], psi[n][j] = Omega * f'(theta[n][j]),
* delta_w[n][j][k] = lambda * a[n][j] * psi[n+1][k].
*/
            for (n = hiddenLast; n >= HIDDEN_SECOND; --n)
            {
               for (j = 0; j < layerSizes[n]; ++j)
               {
                  double omega = 0.0;
                  for (k = 0; k < layerSizes[n + 1]; ++k)
                  {
                     omega += psi[n + 1][k] * w[n][j][k];
                     w[n][j][k] += lambda * a[n][j] * psi[n + 1][k];
                  }
                  psi[n][j] = omega * activationDerivative(theta[n][j]);
               }
            } // for (n = hiddenLast; n >= HIDDEN_SECOND; --n)

/*
* HiddenFirst: Omega = sum_k psi[n+1][k] * w[n][j][k] (read before update), psiJ = Omega * f'(theta[n][j]),
* delta_w[n][j][k] = lambda * a[n][j] * psi[n+1][k], delta_w[n-1][m][j] = lambda * a[n-1][m] * psiJ.
*/
            n = HIDDEN_FIRST;
            for (j = 0; j < layerSizes[n]; ++j)
            {
               double omega = 0.0;
               for (k = 0; k < layerSizes[n + 1]; ++k)
               {
                  omega += psi[n + 1][k] * w[n][j][k];
                  w[n][j][k] += lambda * a[n][j] * psi[n + 1][k];
               }
               double psiJ = omega * activationDerivative(theta[n][j]);

               for (m = 0; m < layerSizes[n - 1]; ++m)
               {
                  w[n - 1][m][j] += lambda * a[n - 1][m] * psiJ;
               }
            } // for (j = 0; j < layerSizes[n]; ++j)

         } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

         avgError = totalError / (2.0 * (double) numCases);
         iteration++;

         if ((keepAlive != 0) && ((iteration % keepAlive) == 0))
            System.out.printf("Iteration %d, Error = %f\n", iteration, avgError);

/*
 * Per-iteration checkpoint dump. Useful when training images with occasional loss
 * spikes (recover from the last good iter). Disabled for normal boolean and finger
 * tests — re-enable by uncommenting if doing spike-prone image training again.
 *
 *       if (saveFile != null && saveFile.length() > 0 && (iteration % 100) == 0)
 *       {
 *          String origSaveFile = saveFile;
 *          int dotIdx = saveFile.lastIndexOf('.');
 *          String stem = (dotIdx > 0) ? saveFile.substring(0, dotIdx) : saveFile;
 *          String ext  = (dotIdx > 0) ? saveFile.substring(dotIdx)    : "";
 *          saveFile = stem + "_iter" + iteration + ext;
 *          saveWeights();
 *          saveFile = origSaveFile;
 *       }
 */

      } // while (iteration < maxIterations && avgError >= errorThreshold)

      trainIterations = iteration;
      trainError = avgError;
   } // train()

/*
* Reports training exit info: reason, iterations, final error, and time.
*/
   public void reportTrainingResults()
   {
      System.out.println("\nTraining Exit Information:");

      if (trainIterations >= maxIterations)
      {
         System.out.println("Reason: Max iterations reached.");
      }

      if (trainError < errorThreshold)
      {
         System.out.println("Reason: Error threshold reached.");
      }

      System.out.println("Iterations: " + trainIterations);
      System.out.println("Average Error: " + trainError);
      System.out.println("Time to train: " + trainTimeMs + " ms");
   } // reportTrainingResults()

/*
* Sets input activations a[INPUT_LAYER][m] from the given case index.
*/
   public void populateInput(int caseIndex)
   {
      for (int m = 0; m < layerSizes[INPUT_LAYER]; ++m)
      {
         a[INPUT_LAYER][m] = inputs[caseIndex][m];
      }
   } // populateInput(int caseIndex)

/*
* Loops over all test cases, setting inputs and running the network.
* Stores each output activation in results[caseIndex][i].
*/
   public void runTestCases()
   {
      int caseIndex;
      int i;
      int outputLayer = numLayers - 1;

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         populateInput(caseIndex);
         runTestCase();

         for (i = 0; i < layerSizes[outputLayer]; ++i)
         {
            results[caseIndex][i] = a[outputLayer][i];
         }
      }
   } // runTestCases()

/*
* Reports run results with expected vs actual outputs for all output nodes.
* Output values are printed to four decimal places.
*/
   public void reportRunResults()
   {
      int caseIndex;
      int i;
      int m;
      int outputLayer = numLayers - 1;
      boolean bareOutputs = !showInputTable && !showTruthTable;
      String fmt = bareOutputs ? "%.4f\t" : "%.4f\t";

      System.out.println();
      if (showInputTable)
      {
         System.out.print("Inputs\t\t");
      }
      if (showTruthTable && isTraining)
      {
         System.out.print("| Expected\t");
      }
      if (!bareOutputs)
      {
         System.out.println("| Actual");
      }

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         if (showInputTable)
         {
            for (m = 0; m < layerSizes[INPUT_LAYER]; ++m)
            {
               System.out.print(inputs[caseIndex][m] + "\t");
            }
         }

         if (showTruthTable && isTraining)
         {
            System.out.print("| ");
            for (i = 0; i < layerSizes[outputLayer]; ++i)
            {
               System.out.print(targets[caseIndex][i] + "\t");
            }
         }

         if (!bareOutputs)
         {
            System.out.print("| ");
         }
         for (i = 0; i < layerSizes[outputLayer]; ++i)
         {
            System.out.printf(fmt, results[caseIndex][i]);
         }
         System.out.println();
      } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

      System.out.println("Time to run: " + runTimeMs + " ms");
   } // reportRunResults()
} // class NLayer