/*
* Author: Tarush Gupta
* Date: April 21, 2026
* Description: A-B-C-D four-activation-layer neural network with backpropagation.
*              Supports configurable layer sizes, random weight initialization or
*              binary weight load/save, and training or running modes selected at
*              runtime via a JSON configuration file.
*/
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;

public class ABCD
{
   public static final int INPUTLAYER = 0;
   public static final int HIDDENLAYER1 = 1;
   public static final int HIDDENLAYER2 = 2;
   public static final int OUTPUTLAYER = 3;
   public static final int activationLayerCount = 4;
   public static final int lastHiddenLayer = 2;

/*
* Network configuration
*/
   public int inputLayer;
   public int hiddenLayer1;
   public int hiddenLayer2;
   public int outputLayer;
   public double lambda;
   public double low;
   public double high;
   public int maxIterations;
   public double errorThreshold;
   public boolean useRandomWeights;
   public int numCases;

/*
* Network arrays.
*/
   public double[][] a;
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
   public int trainIterations;
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

      inputLayer = Integer.parseInt(getJsonValue(json, "inputLayer"));
      hiddenLayer1 = Integer.parseInt(getJsonValue(json, "hiddenLayer1"));
      hiddenLayer2 = Integer.parseInt(getJsonValue(json, "hiddenLayer2"));
      outputLayer = Integer.parseInt(getJsonValue(json, "outputLayer"));
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
      shouldRun = Boolean.parseBoolean(getJsonValue(json, "shouldRun"));
      showInputTable = Boolean.parseBoolean(getJsonValue(json, "showInputTable"));
      showTruthTable = Boolean.parseBoolean(getJsonValue(json, "showTruthTable"));
      loadFile = getJsonValue(json, "loadFile");
      saveFile = getJsonValue(json, "saveFile");
   } // setConfig(String configFile)

/*
* Extracts the raw value string for the given key from a JSON string.
* Searches for "key": and returns trimmed text up to the next comma or closing brace.
* Returns empty string if the key is not found.
*/
   private String getJsonValue(String json, String key)
   {
      String result = "";
      int keyIdx = json.indexOf("\"" + key + "\"");

      if (keyIdx != -1)
      {
         int colonIdx = json.indexOf(":", keyIdx);
         int commaIdx = json.indexOf(",", colonIdx);
         int braceIdx = json.indexOf("}", colonIdx);
         int endIdx = (commaIdx != -1 && commaIdx < braceIdx) ? commaIdx : braceIdx;
         result = json.substring(colonIdx + 1, endIdx).trim().replace("\"", "");
      }

      return result;
   } // getJsonValue(String json, String key)

/*
* Prints current network config including file names and training parameters when applicable.
*/
   public void echoConfig()
   {
      System.out.println("Config File: " + configFile);
      System.out.println("Network Configuration: " + inputLayer + "-" + hiddenLayer1
         + "-" + hiddenLayer2 + "-" + outputLayer);
      System.out.println("Number of Test Cases: " + numCases);
      System.out.println("Input File: " + inputFile);
      System.out.println("Mode: " + (isTraining ? "Training" : "Running"));
      System.out.println("Running: " + shouldRun);

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
      a = new double[activationLayerCount][];
      a[INPUTLAYER] = new double[inputLayer];
      a[HIDDENLAYER1] = new double[hiddenLayer1];
      a[HIDDENLAYER2] = new double[hiddenLayer2];
      a[OUTPUTLAYER] = new double[outputLayer];

      w = new double[activationLayerCount - 1][][];
      w[INPUTLAYER] = new double[inputLayer][hiddenLayer1];
      w[HIDDENLAYER1] = new double[hiddenLayer1][hiddenLayer2];
      w[HIDDENLAYER2] = new double[hiddenLayer2][outputLayer];

      inputs = new double[numCases][inputLayer];

      if (!isTraining || shouldRun)
      {
         results = new double[numCases][outputLayer];
      }

      if (isTraining)
      {
         targets = new double[numCases][outputLayer];
         theta = new double[activationLayerCount - 1][];
         theta[HIDDENLAYER1] = new double[hiddenLayer1];
         theta[HIDDENLAYER2] = new double[hiddenLayer2];

         psi = new double[activationLayerCount][];
         psi[HIDDENLAYER2] = new double[hiddenLayer2];
         psi[OUTPUTLAYER] = new double[outputLayer];
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
         for (int m = 0; m < inputLayer; ++m)
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

      for (int c = 0; c < numCases; ++c)
      {
         String[] tokens = br.readLine().trim().split("\\s+");
         for (int i = 0; i < outputLayer; ++i)
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
         useRandomWeights();
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
         for (int m = 0; m < inputLayer; ++m)
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
      System.out.println("\nTruth Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int m = 0; m < inputLayer; ++m)
         {
            System.out.print(inputs[c][m] + "\t");
         }
         System.out.print("| ");
         for (int i = 0; i < outputLayer; ++i)
         {
            System.out.print(targets[c][i] + "\t");
         }
         System.out.println();
      } // for (int c = 0; c < numCases; ++c)
   } // printTruthTable()

/*
* Initializes all weights randomly using the randomize function.
*/
   public void useRandomWeights()
   {
      int n;
      int m;
      int k;
      int j;
      int i;

      n = HIDDENLAYER1; // mk layer: w[n-1] = w[INPUTLAYER]
      for (m = 0; m < inputLayer; ++m)
      {
         for (k = 0; k < hiddenLayer1; ++k)
         {
            w[n-1][m][k] = randomize(low, high);
         }
      }

      n = HIDDENLAYER2; // kj layer: w[n-1] = w[HIDDENLAYER1]
      for (k = 0; k < hiddenLayer1; ++k)
      {
         for (j = 0; j < hiddenLayer2; ++j)
         {
            w[n-1][k][j] = randomize(low, high);
         }
      }

      n = OUTPUTLAYER; // ji layer: w[n-1] = w[HIDDENLAYER2]
      for (j = 0; j < hiddenLayer2; ++j)
      {
         for (i = 0; i < outputLayer; ++i)
         {
            w[n-1][j][i] = randomize(low, high);
         }
      }
   } // useRandomWeights()

/*
* Uses the weights from the loaded file.
*/
   public void useLoadedWeights()
   {
      try
      {
         DataInputStream in = new DataInputStream(new FileInputStream(loadFile));

         int fileA = in.readInt();
         int fileB = in.readInt();
         int fileC = in.readInt();
         int fileD = in.readInt();

         if (fileA == inputLayer && fileB == hiddenLayer1
            && fileC == hiddenLayer2 && fileD == outputLayer)
         {
            int n;
            int m;
            int k;
            int j;
            int i;

            n = HIDDENLAYER1; // mk weights: w[n-1] = w[INPUTLAYER]
            for (m = 0; m < inputLayer; ++m)
            {
               for (k = 0; k < hiddenLayer1; ++k)
               {
                  w[n-1][m][k] = in.readDouble();
               }
            }

            n = HIDDENLAYER2; // kj weights: w[n-1] = w[HIDDENLAYER1]
            for (k = 0; k < hiddenLayer1; ++k)
            {
               for (j = 0; j < hiddenLayer2; ++j)
               {
                  w[n-1][k][j] = in.readDouble();
               }
            }

            n = OUTPUTLAYER; // ji weights: w[n-1] = w[HIDDENLAYER2]
            for (j = 0; j < hiddenLayer2; ++j)
            {
               for (i = 0; i < outputLayer; ++i)
               {
                  w[n-1][j][i] = in.readDouble();
               }
            }
         } // if (fileA == inputLayer && fileB == hiddenLayer1 ...)
         else
         {
            System.out.println("Error: weight file config (" + fileA + "-" + fileB
               + "-" + fileC + "-" + fileD + ") does not match network config ("
               + inputLayer + "-" + hiddenLayer1 + "-" + hiddenLayer2 + "-" + outputLayer + ").");
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

         out.writeInt(inputLayer);
         out.writeInt(hiddenLayer1);
         out.writeInt(hiddenLayer2);
         out.writeInt(outputLayer);

         int n;
         int m;
         int k;
         int j;
         int i;

         n = HIDDENLAYER1; // mk weights: w[n-1] = w[INPUTLAYER]
         for (m = 0; m < inputLayer; ++m)
         {
            for (k = 0; k < hiddenLayer1; ++k)
            {
               out.writeDouble(w[n-1][m][k]);
            }
         }

         n = HIDDENLAYER2; // kj weights: w[n-1] = w[HIDDENLAYER1]
         for (k = 0; k < hiddenLayer1; ++k)
         {
            for (j = 0; j < hiddenLayer2; ++j)
            {
               out.writeDouble(w[n-1][k][j]);
            }
         }

         n = OUTPUTLAYER; // ji weights: w[n-1] = w[HIDDENLAYER2]
         for (j = 0; j < hiddenLayer2; ++j)
         {
            for (i = 0; i < outputLayer; ++i)
            {
               out.writeDouble(w[n-1][j][i]);
            }
         }

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
      double e = Math.exp(eps * 2.0 * x);
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
* Runs the network forward on pre-populated input activations for the given
* case. Forward pass: mk layer -> kj layer -> ji layer.
* Computes psi[OUTPUTLAYER] for use in the backward pass.
*/
   public double runTrainCase(int caseIndex)
   {
      int n;
      int m;
      int k;
      int j;
      int i;

/*
* mk layer: n = HIDDENLAYER1.
*/
      n = HIDDENLAYER1;
      for (k = 0; k < hiddenLayer1; ++k)
      {
         theta[n][k] = 0.0;
         for (m = 0; m < inputLayer; ++m)
         {
            theta[n][k] += a[n-1][m] * w[n-1][m][k];
         }
         a[n][k] = activation(theta[n][k]);
      } // for (k = 0; k < hiddenLayer1; ++k)

/*
* kj layer: n = HIDDENLAYER2.
*/
      n = HIDDENLAYER2;
      for (j = 0; j < hiddenLayer2; ++j)
      {
         theta[n][j] = 0.0;
         for (k = 0; k < hiddenLayer1; ++k)
         {
            theta[n][j] += a[n-1][k] * w[n-1][k][j];
         }
         a[n][j] = activation(theta[n][j]);
      } // for (j = 0; j < hiddenLayer2; ++j)

      double twiceError = 0.0;

/*
* ji layer: n = OUTPUTLAYER.
* omega_i = T_i - F_i, psi_i = omega_i * f'(theta_i), 2E += omega_i^2.
*/
      n = OUTPUTLAYER;
      for (i = 0; i < outputLayer; ++i)
      {
         double thetaI = 0.0;
         for (j = 0; j < hiddenLayer2; ++j)
         {
            thetaI += a[n-1][j] * w[n-1][j][i];
         }
         a[n][i]       = activation(thetaI);
         double omegaI = targets[caseIndex][i] - a[n][i];
         psi[n][i]     = omegaI * activationDerivative(thetaI);
         twiceError   += omegaI * omegaI;
      } // for (i = 0; i < outputLayer; ++i)

      return twiceError;
   } // runTrainCase(int caseIndex)

/*
* Runs the network forward on pre-populated input activations.
* Forward pass: mk layer -> kj layer -> ji layer.
*/
   public void runTestCase()
   {
      int n;
      int m;
      int k;
      int j;
      int i;

/*
* mk layer: n = HIDDENLAYER1.
*/
      n = HIDDENLAYER1;
      for (k = 0; k < hiddenLayer1; ++k)
      {
         double thetaAcc = 0.0;
         for (m = 0; m < inputLayer; ++m)
         {
            thetaAcc += a[n-1][m] * w[n-1][m][k];
         }
         a[n][k] = activation(thetaAcc);
      }

/*
* kj layer: n = HIDDENLAYER2.
*/
      n = HIDDENLAYER2;
      for (j = 0; j < hiddenLayer2; ++j)
      {
         double thetaAcc = 0.0;
         for (k = 0; k < hiddenLayer1; ++k)
         {
            thetaAcc += a[n-1][k] * w[n-1][k][j];
         }
         a[n][j] = activation(thetaAcc);
      }

/*
* ji layer: n = OUTPUTLAYER.
*/
      n = OUTPUTLAYER;
      for (i = 0; i < outputLayer; ++i)
      {
         double thetaAcc = 0.0;
         for (j = 0; j < hiddenLayer2; ++j)
         {
            thetaAcc += a[n-1][j] * w[n-1][j][i];
         }
         a[n][i] = activation(thetaAcc);
      }
   } // runTestCase()

/*
* Trains the network using steepest descent across all training cases.
* Stores final iteration count and error in trainIterations and trainError.
* Backward pass: ji layer -> kj/mk layers.
*/
   public void train()
   {
      int iteration = 0;
      double avgError = Double.MAX_VALUE;
      int caseIndex;
      int n;
      int m;
      int k;
      int j;
      int i;
      double totalError;

      while (iteration < maxIterations && avgError >= errorThreshold)
      {
         totalError = 0.0;

         for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
         {
            populateInput(caseIndex);
            totalError += runTrainCase(caseIndex);

/*
* ji layer: Omega_j = sum_I psi[n][I] * w[n-1][j][I], Psi[n-1][j] = Omega_j * f'(theta[n-1][j])
* delta_w[n-1][j][i] = lambda * a[n-1][j] * psi[n][i]
*/
            n = OUTPUTLAYER;
            for (j = 0; j < hiddenLayer2; ++j)
            {
               double omegaJ = 0.0;
               for (i = 0; i < outputLayer; ++i)
               {
                  omegaJ       += psi[n][i] * w[n-1][j][i];
                  w[n-1][j][i] += lambda * a[n-1][j] * psi[n][i];
               }
               psi[n-1][j] = omegaJ * activationDerivative(theta[n-1][j]);
            } // for (j = 0; j < hiddenLayer2; ++j)

/*
* kj/mk layers: Omega_k = sum_J psi[n][J] * w[n-1][k][J], psiK = Omega_k * f'(theta[n-1][k])
* delta_w[n-1][k][j] = lambda * a[n-1][k] * psi[n][j], delta_w[n-2][m][k] = lambda * a[n-2][m] * psiK
*/
            n = HIDDENLAYER2;
            for (k = 0; k < hiddenLayer1; ++k)
            {
               double omegaK = 0.0;
               for (j = 0; j < hiddenLayer2; ++j)
               {
                  omegaK       += psi[n][j] * w[n-1][k][j];
                  w[n-1][k][j] += lambda * a[n-1][k] * psi[n][j];
               }
               double psiK = omegaK * activationDerivative(theta[n-1][k]);
               for (m = 0; m < inputLayer; ++m)
               {
                  w[n-2][m][k] += lambda * a[n-2][m] * psiK;
               }
            } // for (k = 0; k < hiddenLayer1; ++k)

         } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

         avgError = totalError / (2.0 * (double) numCases);
         iteration++;
      } // while (iteration < maxIterations && avgError >= errorThreshold)

      trainIterations = iteration;
      trainError      = avgError;
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

      System.out.println("Iterations reached: " + trainIterations);
      System.out.println("Average Error reached: " + trainError);
      System.out.println("Time to train: " + trainTimeMs + " ms");
   } // reportTrainingResults()

/*
* Loops over all test cases, setting inputs and running the network.
* Stores each output a[OUTPUTLAYER][i] in results[caseIndex][i].
*/
   public void runTestCases()
   {
      int caseIndex;
      int i;

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         populateInput(caseIndex);
         runTestCase();

         for (i = 0; i < outputLayer; ++i)
         {
            results[caseIndex][i] = a[OUTPUTLAYER][i];
         }
      }
   } // runTestCases()

/*
* Sets input activations a[INPUTLAYER][m] from the given case index.
*/
   public void populateInput(int caseIndex)
   {
      for (int m = 0; m < inputLayer; ++m)
      {
         a[INPUTLAYER][m] = inputs[caseIndex][m];
      }
   } // populateInput(int caseIndex)

/*
* Reports run results with expected vs actual outputs for all output nodes.
* Output values are printed to four decimal places.
*/
   public void reportRunResults()
   {
      int caseIndex;
      int i;
      int m;

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
            for (m = 0; m < inputLayer; ++m)
            {
               System.out.print(inputs[caseIndex][m] + "\t");
            }
         }

         if (showTruthTable && isTraining)
         {
            System.out.print("| ");
            for (i = 0; i < outputLayer; ++i)
            {
               System.out.print(targets[caseIndex][i] + "\t");
            }
         }

         System.out.print("| ");
         for (i = 0; i < outputLayer; ++i)
         {
            System.out.printf("%.4f\t", results[caseIndex][i]);
         }
         System.out.println();
      } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

      System.out.println("Time to run: " + runTimeMs + " ms");
   } // reportRunResults()
} // class ABCD
