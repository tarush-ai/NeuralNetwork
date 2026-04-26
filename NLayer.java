/*
* Author: Tarush Gupta
* Date: April 26, 2026
* Description: N-activation-layer neural network with backpropagation.
*              Generalizes to any number of layers with any number of nodes per layer.
*              Layer sizes, training parameters, and file paths are all selected at
*              runtime via a JSON configuration file with a comma-separated "layers"
*              field (e.g., "2,20,3"). Supports random weight initialization or binary
*              weight load/save, and training or running modes.
*
* Table of Contents:
*    public void setConfig(String configFile)
*    private String getJsonValue(String json, String key)
*    public void echoConfig()
*    public void allocateMemory()
*    private void loadInputs(String filename)
*    private void loadTruthTable(String filename)
*    public void populateArrays()
*    public void printInputTable()
*    public void printTruthTable()
*    public void randomizeWeights()
*    public void useLoadedWeights()
*    public void saveWeights()
*    public double randomize(double low, double high)
*    public double activation(double x)
*    public double activationDerivative(double x)
*    public double sigmoid(double x)
*    public double sigmoidDerivative(double x)
*    public double tanh(double x)
*    public double tanhDerivative(double x)
*    public double linear(double x)
*    public double linearDerivative(double x)
*    public double runTrainCase(int caseIndex)
*    public void runTestCase()
*    public void train()
*    public void reportTrainingResults()
*    public void populateInput(int caseIndex)
*    public void runTestCases()
*    public void reportRunResults()
*/
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;

public class NLayer
{
/*
* Network geometry: number of activation layers and size of each.
*/
   public int    numLayers;
   public int[]  layerSizes;

/*
* Training hyperparameters.
*/
   public double  lambda;
   public double  low;
   public double  high;
   public int     maxIterations;
   public double  errorThreshold;
   public boolean useRandomWeights;
   public int     numCases;
   public int     keepAlive;

/* 
* Network arrays: activations, weights, inputs.
*/
   public double[][]  a;
   public double[][][] w;
   public double[][]  inputs;

/*
* Run results.
*/
   public double[][] results;

/*
* Training-only arrays: targets, pre-activations, error signals.
*/
   public double[][] targets;
   public double[][] theta;
   public double[][] psi;

/*
* Timing and training outcome.
*/
   public int  trainIterations;
   public double trainError;
   public long trainTimeMs;
   public long runTimeMs;

/*
* Runtime flags and file paths.
*/
   public boolean isTraining;
   public boolean shouldRun;
   public boolean showInputTable;
   public boolean showTruthTable;
   public String  loadFile;
   public String  saveFile;
   public String  inputFile;
   public String  truthTableFile;
   public String  configFile;

/*
* Activation function. Change this line (and activationDerivative below) to switch:
* sigmoid(x), tanh(x), or linear(x).
*/
   public double activation(double x)
   {
      return sigmoid(x);
   } // activation(double x)

/*
* Activation function derivative. Must match the choice in activation() above:
* sigmoidDerivative(x), tanhDerivative(x), or linearDerivative(x).
*/
   public double activationDerivative(double x)
   {
      return sigmoidDerivative(x);
   } // activationDerivative(double x)

/*
* End of user configuration section.
*/

/*
* Reads all network parameters from the given JSON config file.
* The "layers" key must be a comma-separated string of layer sizes (e.g., "2,20,3").
*/
   public void setConfig(String configFile) throws Exception
   {
      this.configFile = configFile;

      StringBuilder sb     = new StringBuilder();
      BufferedReader reader = new BufferedReader(new FileReader(configFile));
      String line;
      while ((line = reader.readLine()) != null)
      {
         sb.append(line);
      }
      reader.close();
      String json = sb.toString();

      String   layersStr = getJsonValue(json, "layers");
      String[] parts     = layersStr.split(",");
      numLayers          = parts.length;
      layerSizes         = new int[numLayers];

      for (int i = 0; i < numLayers; ++i)
      {
         layerSizes[i] = Integer.parseInt(parts[i].trim());
      }

      String keepAliveStr  = getJsonValue(json, "keepAlive");
      numCases             = Integer.parseInt(getJsonValue(json, "numCases"));
      inputFile            = getJsonValue(json, "inputFile");
      truthTableFile       = getJsonValue(json, "truthTableFile");
      isTraining           = Boolean.parseBoolean(getJsonValue(json, "isTraining"));
      useRandomWeights     = Boolean.parseBoolean(getJsonValue(json, "useRandomWeights"));
      low                  = Double.parseDouble(getJsonValue(json, "low"));
      high                 = Double.parseDouble(getJsonValue(json, "high"));
      lambda               = Double.parseDouble(getJsonValue(json, "lambda"));
      maxIterations        = Integer.parseInt(getJsonValue(json, "maxIterations"));
      errorThreshold       = Double.parseDouble(getJsonValue(json, "errorThreshold"));
      keepAlive            = keepAliveStr.isEmpty() ? 0 : Integer.parseInt(keepAliveStr);
      shouldRun            = Boolean.parseBoolean(getJsonValue(json, "shouldRun"));
      showInputTable       = Boolean.parseBoolean(getJsonValue(json, "showInputTable"));
      showTruthTable       = Boolean.parseBoolean(getJsonValue(json, "showTruthTable"));
      loadFile             = getJsonValue(json, "loadFile");
      saveFile             = getJsonValue(json, "saveFile");
   } // setConfig(String configFile)

/*
* Extracts the value string for the given key from a flat JSON string.
* For quoted values, returns content between the surrounding quotes.
* For unquoted values, returns trimmed text up to the next comma or closing brace.
* Returns empty string if the key is not found.
*/
   private String getJsonValue(String json, String key)
   {
      String result = "";
      int    keyIdx = json.indexOf("\"" + key + "\"");

      if (keyIdx != -1)
      {
         int    colonIdx = json.indexOf(":", keyIdx);
         String rest     = json.substring(colonIdx + 1).trim();

         if (rest.charAt(0) == '"')
         {
            int openQuote  = json.indexOf('"', colonIdx + 1);
            int closeQuote = json.indexOf('"', openQuote + 1);
            result = json.substring(openQuote + 1, closeQuote);
         }
         else
         {
            int commaIdx = json.indexOf(",", colonIdx);
            int braceIdx = json.indexOf("}", colonIdx);
            int endIdx   = (commaIdx != -1 && commaIdx < braceIdx) ? commaIdx : braceIdx;
            result = json.substring(colonIdx + 1, endIdx).trim();
         }
      } // if (keyIdx != -1)

      return result;
   } // getJsonValue(String json, String key)

/*
* Prints current network configuration including all layer sizes, file names,
* and training parameters when applicable.
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
* Allocates all network arrays. Training-only arrays (theta, psi, targets) are
* allocated only when isTraining is true. results is allocated only when needed
* for a run pass.
*/
   public void allocateMemory()
   {
      int n;
      int outputLayer = numLayers - 1;

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

      inputs = new double[numCases][layerSizes[0]];

      if (!isTraining || shouldRun)
      {
         results = new double[numCases][layerSizes[outputLayer]];
      }

      if (isTraining)
      {
         targets = new double[numCases][layerSizes[outputLayer]];

         theta = new double[numLayers][];
         for (n = 1; n <= numLayers - 2; ++n)
         {
            theta[n] = new double[layerSizes[n]];
         }

         psi = new double[numLayers][];
         for (n = 2; n < numLayers; ++n)
         {
            psi[n] = new double[layerSizes[n]];
         }
      } // if (isTraining)
   } // allocateMemory()

/*
* Reads binary input activations from filename into the inputs array.
*/
   private void loadInputs(String filename) throws Exception
   {
      DataInputStream in = new DataInputStream(new FileInputStream(filename));

      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < layerSizes[0]; ++m)
         {
            inputs[c][m] = in.readDouble();
         }
      }

      in.close();
   } // loadInputs(String filename)

/*
* Reads a whitespace-delimited truth table from filename into the targets array.
*/
   private void loadTruthTable(String filename) throws Exception
   {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      int outputLayer   = numLayers - 1;

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
* Loads inputs and (if training) targets, then populates weights from random
* values or a saved binary file, as configured.
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
* Prints the raw input table for all test cases.
*/
   public void printInputTable()
   {
      System.out.println("\nInput Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < layerSizes[0]; ++m)
         {
            System.out.print(inputs[c][m] + "\t");
         }
         System.out.println();
      }
   } // printInputTable()

/*
* Prints inputs and targets side by side for all test cases.
*/
   public void printTruthTable()
   {
      int outputLayer = numLayers - 1;

      System.out.println("\nTruth Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < layerSizes[0]; ++m)
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
* Fills all weight matrices with random values drawn from [low, high].
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
* Loads weights from loadFile. Reads numLayers and all layer sizes from the file
* and verifies they match the current network before loading weight values.
*/
   public void useLoadedWeights()
   {
      try
      {
         DataInputStream in          = new DataInputStream(new FileInputStream(loadFile));
         int             fileNumLayers = in.readInt();
         int[]           fileSizes   = new int[fileNumLayers];
         boolean         match;
         int             n;
         int             j;
         int             k;

         for (n = 0; n < fileNumLayers; ++n)
         {
            fileSizes[n] = in.readInt();
         }

         match = (fileNumLayers == numLayers);
         for (n = 0; n < numLayers && match; ++n)
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
* Saves weights to saveFile. Writes numLayers, all layer sizes, then all weights.
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
* Returns a random double uniformly drawn from [low, high].
*/
   public double randomize(double low, double high)
   {
      return low + Math.random() * (high - low);
   } // randomize(double low, double high)

/*
* Sigmoid activation: f(x) = 1 / (1 + e^-x).
*/
   public double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // sigmoid(double x)

/*
* Sigmoid derivative: f'(x) = f(x) * (1 - f(x)).
*/
   public double sigmoidDerivative(double x)
   {
      double val = activation(x);
      return val * (1.0 - val);
   } // sigmoidDerivative(double x)

/*
* Hyperbolic tangent using ε(x) = (x < 0) ? 1 : -1.
* exp(ε * 2x) is computed once and reused in numerator and denominator.
*/
   public double tanh(double x)
   {
      double eps = (x < 0) ? 1.0 : -1.0;
      double e   = Math.exp(eps * 2.0 * x);
      return eps * (e - 1.0) / (e + 1.0);
   } // tanh(double x)

/*
* Hyperbolic tangent derivative: f'(x) = 1 - f(x)^2.
*/
   public double tanhDerivative(double x)
   {
      double val = tanh(x);
      return 1.0 - val * val;
   } // tanhDerivative(double x)

/*
* Linear activation: f(x) = x.
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
* Forward pass for training. Blue box computes theta and a for hidden layers
* (α: 1 to numLayers-2). Green box computes a, ψ, and accumulates E for the
* output layer. Returns twice the total squared error for the given case.
*/
   public double runTrainCase(int caseIndex)
   {
      int    n;
      int    j;
      int    k;
      int    outputLayer = numLayers - 1;
      double twiceError  = 0.0;

/*
* Blue box: α from HiddenFirst to HiddenLast.
*/
      for (n = 1; n <= numLayers - 2; ++n)
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
      } // for (n = 1; n <= numLayers - 2; ++n)

/*
* Green box: α = output layer. Compute a, ψ, and accumulate E.
*/
      n = outputLayer;
      for (j = 0; j < layerSizes[n]; ++j)
      {
         double thetaJ = 0.0;
         for (k = 0; k < layerSizes[n - 1]; ++k)
         {
            thetaJ += a[n - 1][k] * w[n - 1][k][j];
         }
         a[n][j]       = activation(thetaJ);
         double omegaJ = targets[caseIndex][j] - a[n][j];
         psi[n][j]     = omegaJ * activationDerivative(thetaJ);
         twiceError   += omegaJ * omegaJ;
      } // for (j = 0; j < layerSizes[n]; ++j)

      return twiceError;
   } // runTrainCase(int caseIndex)

/*
* Forward pass for inference only (no theta or psi storage).
* Red box: all layers α from 1 to numLayers-1.
*/
   public void runTestCase()
   {
      int n;
      int j;
      int k;

      for (n = 1; n < numLayers; ++n)
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
      } // for (n = 1; n < numLayers; ++n)
   } // runTestCase()

/*
* Steepest-descent training across all cases until error threshold or max
* iterations is reached.
*
* Purple box (α: HiddenLast downto HiddenSecond):
*   n indexes the SOURCE activation layer. w[n] connects layer n to n+1.
*   Computes Ω(n,j) and Ψ(n,j) using pre-update weights, then updates w[n].
*
* Orange box (α = HiddenFirst = 1):
*   Part 1 — update w[1] using a[1] and ψ(2).
*   Part 2 — compute Ψ(1) using post-update w[1] (O(λ²) perturbation, intentional
*             per design: collapses two m-loops into one).
*   Part 3 — update w[0] using a[0] and Ψ(1).
*/
   public void train()
   {
      int    iteration = 0;
      double avgError  = Double.MAX_VALUE;
      int    caseIndex;
      int    n;
      int    j;
      int    k;
      int    m;
      double totalError;

      while (iteration < maxIterations && avgError >= errorThreshold)
      {
         totalError = 0.0;

         for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
         {
            populateInput(caseIndex);
            totalError += runTrainCase(caseIndex);

/*
* Purple box: α from HiddenLast downto HiddenSecond.
* n is the source activation layer index for weight layer w[n].
*/
            for (n = numLayers - 2; n >= 2; --n)
            {
               for (j = 0; j < layerSizes[n]; ++j)
               {
                  double omega = 0.0;
                  for (k = 0; k < layerSizes[n + 1]; ++k)
                  {
                     omega      += psi[n + 1][k] * w[n][j][k];
                     w[n][j][k] += lambda * a[n][j] * psi[n + 1][k];
                  }
                  psi[n][j] = omega * activationDerivative(theta[n][j]);
               }
            } // for (n = numLayers - 2; n >= 2; --n)

/*
* Orange box: α = HiddenFirst = 1.
* Part 1: update w[1][j][k].
* Part 2: compute Ψ(1,j) using post-update w[1].
* Part 3: update w[0][m][j] using Ψ(1,j).
*/
            n = 1;
            for (j = 0; j < layerSizes[n]; ++j)
            {
               for (k = 0; k < layerSizes[n + 1]; ++k)
               {
                  w[n][j][k] += lambda * a[n][j] * psi[n + 1][k];
               }

               double omega = 0.0;
               for (k = 0; k < layerSizes[n + 1]; ++k)
               {
                  omega += psi[n + 1][k] * w[n][j][k];
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

      } // while (iteration < maxIterations && avgError >= errorThreshold)

      trainIterations = iteration;
      trainError      = avgError;
   } // train()

/*
* Reports training exit reason, final iteration count, average error, and elapsed time.
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
* Sets input activations a[0][m] from inputs[caseIndex].
*/
   public void populateInput(int caseIndex)
   {
      for (int m = 0; m < layerSizes[0]; ++m)
      {
         a[0][m] = inputs[caseIndex][m];
      }
   } // populateInput(int caseIndex)

/*
* Runs all test cases and stores output activations in results[][].
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
* Prints inputs, expected outputs (when training), and actual outputs for every case.
*/
   public void reportRunResults()
   {
      int caseIndex;
      int i;
      int m;
      int outputLayer = numLayers - 1;

      if (showInputTable)
      {
         System.out.print("\nInputs\t\t");
      }
      if (showTruthTable && targets != null)
      {
         System.out.print("| Expected\t");
      }
      System.out.println("| Actual");

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         if (showInputTable)
         {
            for (m = 0; m < layerSizes[0]; ++m)
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

         System.out.print("| ");
         for (i = 0; i < layerSizes[outputLayer]; ++i)
         {
            System.out.printf("%.4f\t", results[caseIndex][i]);
         }
         System.out.println();
      } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

      System.out.println("Time to run: " + runTimeMs + " ms");
   } // reportRunResults()
} // class NLayer
