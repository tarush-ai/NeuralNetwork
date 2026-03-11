/*
* Author: Tarush Gupta
* Date: March 13, 2026
* Description: A-B-C Neural Network with gradient descent training.
*              Generalizes AB1 to support multiple output activations F[i].
*              Supports configurable layers, rand/fixed init, training/running modes,
*              and binary weight save/load.
*/
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class ABC
{
/*
* Network config
*/
   public int inputLayer;
   public int hiddenLayer;
   public int outputLayer;
   public double lambda;
   public double low;
   public double high;
   public int maxIterations;
   public double errorThreshold;
   public boolean useRandomWeights;
   public int numCases;

/*
* Network arrays
*/
   public double[] a;
   public double[] h;
   public double[] F;
   public double[][] wKJ;
   public double[][] wJI;
   public double[] thetaJ;
   public double[] thetaI;
   public double[][] inputs;

/*
* Run results 
*/
   public double[][] results;

/*
* Training-only arrays
*/
   public double[][] targets;
   public double[][] deltaWKJ;
   public double[][] deltaWJI;
   public double[] psiI;
   public double[] psiJ;

/*
* Training results
*/
   public int trainIterations;
   public double trainError;
   public long trainTimeMs;
   public long runTimeMs;

   public boolean isTraining;
   public boolean shouldRun;
   public boolean showInputTable;
   public boolean showTruthTable;
   public String loadFile;
   public String saveFile;

/*
* Sets config for the network.
*/
   public void setConfig()
   {
      inputLayer = 2;
      hiddenLayer = 5;
      outputLayer = 3;
      numCases = 4;
      low = -1.5;
      high = 1.5;
      maxIterations = 100000;
      lambda = 0.3;
      errorThreshold = 2e-4;
      useRandomWeights = true;
      isTraining = true;
      shouldRun = true;
      showInputTable = true;
      showTruthTable = true;
/*
      loadFile = "load.bin";
      saveFile = "save.bin"; 
*/
   } // setConfig()

/*
* Prints current network config including training parameters when applicable.
*/
   public void echoConfig()
   {
      System.out.println("Network Configuration: " + inputLayer + "-" + hiddenLayer + "-" + outputLayer);
      System.out.println("Number of Test Cases: " + numCases);
      System.out.println("Weight Initialization: " + (useRandomWeights ? "Random" : "Fixed"));
      System.out.println("Run After Training: " + shouldRun);
      System.out.println("Mode: " + (isTraining ? "Training" : "Running"));

      if (isTraining)
      {
         System.out.println("\nTraining Parameters:");
         System.out.println("Random Number Range: [" + low + ", " + high + "]");
         System.out.println("Max Iterations: " + maxIterations);
         System.out.println("Error Threshold: " + errorThreshold);
         System.out.println("Lambda: " + lambda);
      }
   } // echoConfig()

/*
* Allocates memory for all network arrays. Training-only arrays
* are only allocated when the network is in training mode.
*/
   public void allocateMemory()
   {
      a = new double[inputLayer];
      h = new double[hiddenLayer];
      F = new double[outputLayer];
      wKJ = new double[inputLayer][hiddenLayer];
      wJI = new double[hiddenLayer][outputLayer];
      thetaJ = new double[hiddenLayer];
      thetaI = new double[outputLayer];

      inputs = new double[numCases][inputLayer];
      results = new double[numCases][outputLayer];

      if (isTraining)
      {
         targets = new double[numCases][outputLayer];
         deltaWKJ = new double[inputLayer][hiddenLayer];
         deltaWJI = new double[hiddenLayer][outputLayer];
         psiI = new double[outputLayer];
         psiJ = new double[hiddenLayer];
      }
   } // allocateMemory()

/*
* Populate all of the network arrays based on the type of initialization
* (i.e. random or fixed) and whether the network is training or not.
*/
   public void populateArrays()
   {
      setInputTable();

      if (isTraining)
      {
         setTruthTable();
      }

      if (useRandomWeights)
      {
         useRandomWeights();
      }
      else if ((loadFile != null && saveFile != null) && (loadFile.length() > 0 && saveFile.length() > 0))
      {
         useLoadedWeights();
      }
      else
      {
         useFixedWeights();
      }
   } // populateArrays()

/*
* Sets input values for all relevant test cases, contingent on the function
* that is being optimized, such as AND, OR, XOR, or a[0] XOR a[1] AND a[2]
* for A=3.
*/
   public void setInputTable()
   {
      inputs[0][0] = 0.0; inputs[0][1] = 0.0;
      inputs[1][0] = 0.0; inputs[1][1] = 1.0;
      inputs[2][0] = 1.0; inputs[2][1] = 0.0;
      inputs[3][0] = 1.0; inputs[3][1] = 1.0;

/*
* a[0] XOR a[1] AND a[2] (A=3)
inputs[0][0] = 0.0; inputs[0][1] = 0.0; inputs[0][2] = 0.0;
inputs[1][0] = 0.0; inputs[1][1] = 0.0; inputs[1][2] = 1.0;
inputs[2][0] = 0.0; inputs[2][1] = 1.0; inputs[2][2] = 0.0;
inputs[3][0] = 0.0; inputs[3][1] = 1.0; inputs[3][2] = 1.0;
inputs[4][0] = 1.0; inputs[4][1] = 0.0; inputs[4][2] = 0.0;
inputs[5][0] = 1.0; inputs[5][1] = 0.0; inputs[5][2] = 1.0;
inputs[6][0] = 1.0; inputs[6][1] = 1.0; inputs[6][2] = 0.0;
inputs[7][0] = 1.0; inputs[7][1] = 1.0; inputs[7][2] = 1.0;
*/
   } // setInputTable()

/*
* Prints the input table by iterating through the inputs.
*/
   public void printInputTable()
   {
      System.out.println("\nInput Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int k = 0; k < inputLayer; ++k)
         {
            System.out.print(inputs[c][k] + "\t");
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
         for (int k = 0; k < inputLayer; ++k)
         {
            System.out.print(inputs[c][k] + "\t");
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
      for (int k = 0; k < inputLayer; ++k)
      {
         for (int j = 0; j < hiddenLayer; ++j)
         {
            wKJ[k][j] = randomize(low, high);
         }
      } 
      
      for (int j = 0; j < hiddenLayer; ++j)
      {
         for (int i = 0; i < outputLayer; ++i)
         {
            wJI[j][i] = randomize(low, high);
         }
      }
   } // useRandomWeights()

/*
* Uses fixed weights for 2-2-1 testing.
*/
   public void useFixedWeights()
   {
      if (inputLayer == 2 && hiddenLayer == 2 && outputLayer == 1)
      {
         wKJ[0][0] = 0.76;
         wKJ[0][1] = 0.9;
         wKJ[1][0] = 0.9;
         wKJ[1][1] = 0.76;
         wJI[0][0] = 0.2;
         wJI[1][0] = 0.2;
      }
   } // useFixedWeights()

/*
* Sets the truth table.
*/
   public void setTruthTable()
   {

      targets[0][0] = 0.0; targets[0][1] = 0.0; targets[0][2] = 0.0;
      targets[1][0] = 0.0; targets[1][1] = 1.0; targets[1][2] = 1.0;
      targets[2][0] = 0.0; targets[2][1] = 1.0; targets[2][2] = 1.0;
      targets[3][0] = 1.0; targets[3][1] = 1.0; targets[3][2] = 0.0;

/*
* XOR (C=1)
targets[0][0] = 0.0;
targets[1][0] = 1.0;
targets[2][0] = 1.0;
targets[3][0] = 0.0;
*/

/*
* AND (C=1)
targets[0][0] = 0.0;
targets[1][0] = 0.0;
targets[2][0] = 0.0;
targets[3][0] = 1.0;
*/

/*
* OR (C=1)
targets[0][0] = 0.0;
targets[1][0] = 1.0;
targets[2][0] = 1.0;
targets[3][0] = 1.0;
*/

/*
* AND, OR, XOR (C=3)
targets[0][0] = 0.0; targets[0][1] = 0.0; targets[0][2] = 0.0;
targets[1][0] = 0.0; targets[1][1] = 1.0; targets[1][2] = 1.0;
targets[2][0] = 0.0; targets[2][1] = 1.0; targets[2][2] = 1.0;
targets[3][0] = 1.0; targets[3][1] = 1.0; targets[3][2] = 0.0;
*/

/*
* a[0] XOR a[1] AND a[2] (A=3, C=1)
targets[0][0] = 0.0;
targets[1][0] = 0.0;
targets[2][0] = 0.0;
targets[3][0] = 1.0;
targets[4][0] = 0.0;
targets[5][0] = 1.0;
targets[6][0] = 0.0;
targets[7][0] = 0.0;
*/
   } // setTruthTable()

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

         if (fileA == inputLayer && fileB == hiddenLayer && fileC == outputLayer)
         {
            for (int k = 0; k < inputLayer; ++k)
            {
               for (int j = 0; j < hiddenLayer; ++j)
               {
                  wKJ[k][j] = in.readDouble();
               }
            }

            for (int j = 0; j < hiddenLayer; ++j)
            {
               for (int i = 0; i < outputLayer; ++i)
               {
                  wJI[j][i] = in.readDouble();
               }
            }
         }
         else
         {
            System.out.println("Error: Weight file config (" + fileA + "-" + fileB + "-" + fileC
               + ") does not match network config ("
               + inputLayer + "-" + hiddenLayer + "-" + outputLayer + ").");
         } // if (fileA == inputLayer && fileB == hiddenLayer && fileC == outputLayer)

         in.close();
      } // try
      catch (Exception e)
      {
         System.out.println("Error loading weights: " + e.getMessage());
      }
   } // useLoadedWeights(String filename)

/**
 * Saves the weights to the relevant file.
 */
   public void saveWeights()
   {
      try
      {
         DataOutputStream out = new DataOutputStream(new FileOutputStream(saveFile));

         out.writeInt(inputLayer);
         out.writeInt(hiddenLayer);
         out.writeInt(outputLayer);

         for (int k = 0; k < inputLayer; ++k)
         {
            for (int j = 0; j < hiddenLayer; ++j)
            {
               out.writeDouble(wKJ[k][j]);
            }
         }

         for (int j = 0; j < hiddenLayer; ++j)
         {
            for (int i = 0; i < outputLayer; ++i)
            {
               out.writeDouble(wJI[j][i]);
            }
         }

         out.close();
      }
      catch (Exception e)
      {
         e.printStackTrace();
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
* Generalizable activation function. User can specify this to be
* sigmoid or linear.
*/
   public double activation(double x)
   {
      return sigmoid(x);
   }

/*
* Generalizable activation function derivative. User can specify this
* to be the derivative of sigmoid or linear.
*/
   public double activationDerivative(double num)
   {
      return sigmoidDerivative(num);
   }

/*
* Sigmoid activation: f(x) = 1 / (1 + e^-x). Uses Math.exp().
*/
   public double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // sigmoid(double x)

/*
* Sigmoid activation derivative: f'(x) = f(x) * (1 - f(x))
*/
   public double sigmoidDerivative(double x)
   {
      double s = sigmoid(x);
      return s * (1.0 - s);
   }

/*
* Linear activation: f(x) = x
*/
   public double linear(double x)
   {
      return x;
   }

/*
* Linear activation derivative: f'(x) = 1
*/
   public double linearDerivative(double x)
   {
      return 1.0;
   }

/*
* Sets input activations from the given training case and runs the
* network forward. Forward pass: theta_j -> h_j -> theta_i -> F_i
*/
   public void runTrainCase(int caseIndex)
   {
      int i;
      int j;
      int k;

      populateInput(caseIndex);

/*
* Hidden layer: theta_j = sum_K a_K * w_kj, h_j = f(theta_j)
*/
      for (j = 0; j < hiddenLayer; ++j)
      {
         thetaJ[j] = 0.0;
         for (k = 0; k < inputLayer; ++k)
         {
            thetaJ[j] += a[k] * wKJ[k][j];
         }
         h[j] = activation(thetaJ[j]);
      }

/*
* Output layer: theta_i = sum_J h_J * w_ji, F_i = f(theta_i)
*/
      for (i = 0; i < outputLayer; ++i)
      {
         thetaI[i] = 0.0;
         for (j = 0; j < hiddenLayer; ++j)
         {
            thetaI[i] += h[j] * wJI[j][i];
         }
         F[i] = activation(thetaI[i]);
      }
   } // runTrainCase(int caseIndex)

/*
* Sets input activations from the given test case and runs the
* network forward. Forward pass: theta_j -> h_j -> theta_i -> F_i
*/
   public void runTestCase(int caseIndex)
   {
      int i;
      int j;
      int k;

      populateInput(caseIndex);

/*
* Hidden layer: theta_j = sum_K a_K * w_kj, h_j = f(theta_j)
*/
      for (j = 0; j < hiddenLayer; ++j)
      {
         thetaJ[j] = 0.0;
         for (k = 0; k < inputLayer; ++k)
         {
            thetaJ[j] += a[k] * wKJ[k][j];
         }
         h[j] = activation(thetaJ[j]);
      }

/*
* Output layer: theta_i = sum_J h_J * w_ji, F_i = f(theta_i)
*/
      for (i = 0; i < outputLayer; ++i)
      {
         thetaI[i] = 0.0;
         for (j = 0; j < hiddenLayer; ++j)
         {
            thetaI[i] += h[j] * wJI[j][i];
         }
         F[i] = activation(thetaI[i]);
      }
   } // runTestCase(int caseIndex)

/*
* Trains the network using steepest descent across all training cases.
* Stores final iteration count and error in trainIterations and trainError.
* Backward: omega_i -> psi_i -> Omega_j -> Psi_j
* Gradients: delta_w_ji = lambda * h_j * psi_i, delta_w_kj = lambda * a_k * Psi_j
*/
   public void train()
   {
      int iteration = 0;
      double avgError = 1.0;
      int caseIndex;
      int i;
      int j;
      int k;
      double totalError;

      while (iteration < maxIterations && avgError >= errorThreshold)
      {
         totalError = 0.0;

         for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
         {
            runTrainCase(caseIndex);

/*
* omega_i = T_i - F_i, psi_i = omega_i * f'(theta_i)
*/
            for (i = 0; i < outputLayer; ++i)
            {
               double omegaI = targets[caseIndex][i] - F[i];
               totalError += 0.5 * omegaI * omegaI;
               psiI[i] = omegaI * activationDerivative(thetaI[i]);
            }

/*
* Omega_j = sum_I psi_I * w_jI, Psi_j = Omega_j * f'(theta_j)
* delta_w_ji = lambda * h_j * psi_i
* delta_w_kj = lambda * a_k * Psi_j
*/
            for (j = 0; j < hiddenLayer; ++j)
            {
               double omegaJ = 0.0;
               for (i = 0; i < outputLayer; ++i)
               {
                  omegaJ += psiI[i] * wJI[j][i];
               }
               psiJ[j] = omegaJ * activationDerivative(thetaJ[j]);

               for (i = 0; i < outputLayer; ++i)
               {
                  double dE_dWJI = - h[j] * psiI[i];
                  deltaWJI[j][i] = -lambda * dE_dWJI;
               }

               for (k = 0; k < inputLayer; ++k)
               {
                  double dE_dWKJ = - a[k] * psiJ[j];
                  deltaWKJ[k][j] = - lambda * dE_dWKJ;
               }
            } // for (j = 0; j < hiddenLayer; ++j)

/*
* Apply weight changes: w += delta_w
*/
            for (j = 0; j < hiddenLayer; ++j)
            {
               for (i = 0; i < outputLayer; ++i)
               {
                  wJI[j][i] += deltaWJI[j][i];
               }
               for (k = 0; k < inputLayer; ++k)
               {
                  wKJ[k][j] += deltaWKJ[k][j];
               }
            } // for (j = 0; j < hiddenLayer; ++j)
         } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

         avgError = totalError / (double) numCases;
         iteration++;
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
      System.out.println("Iterations reached: " + trainIterations);
      System.out.println("Average Error reached: " + trainError);
      System.out.println("Time to train: " + trainTimeMs + " ms");
   } // reportTrainingResults()

/*
* Loops over all test cases, setting inputs and running the network.
* Stores each output F[i] in results[caseIndex][i].
*/
   public void runTrainingCases()
   {
      int caseIndex;
      int i;

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         runTrainCase(caseIndex);
         for (i = 0; i < outputLayer; ++i)
         {
            results[caseIndex][i] = F[i];
         }
      }
   } // runTrainingCases()

/*
* Loops over all test cases, setting inputs and running the network.
* Stores each output F[i] in results[caseIndex][i].
*/
   public void runTestCases()
   {
      int caseIndex;
      int i;

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         runTestCase(caseIndex);
         for (i = 0; i < outputLayer; ++i)
         {
            results[caseIndex][i] = F[i];
         }
      }
   } // runTestCases()

/*
* Sets input activations from the given case and runs the network.
*/
   public void populateInput(int caseIndex)
   {
      for (int k = 0; k < inputLayer; ++k)
      {
         a[k] = inputs[caseIndex][k];
      }
   }

/*
* Reports run results with expected vs actual outputs for all output nodes.
* Displays inputs, expected targets, and actual F[i] values per case.
*/
   public void reportRunResults()
   {
      int caseIndex;
      int i;
      int k;

      if (showInputTable)
      {
         System.out.print("\nInputs");
         System.out.print("\t".repeat(2));
      }
      if (showTruthTable && isTraining)
      {
         System.out.print("| Expected\t");
      }
      System.out.println("| Actual");

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         if (showInputTable)
         {
            for (k = 0; k < inputLayer; ++k)
            {
               System.out.print(inputs[caseIndex][k] + "\t");
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
            System.out.print(results[caseIndex][i] + "\t");
         }
         System.out.println();
      } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

      System.out.println("Time to run: " + runTimeMs + " ms");
   } // reportRunResults()
} // class ABC