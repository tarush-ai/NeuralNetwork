import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.PrintWriter;

public class Util
{
   public static void saveInputs(String filename, double[][] inputs) throws Exception
   {
      DataOutputStream out = new DataOutputStream(new FileOutputStream(filename));
      for (double[] row : inputs)
         for (double val : row)
            out.writeDouble(val);
      out.close();
      System.out.println("Saved inputs to: " + filename);
   }


   public static void saveTruthTable(String filename, double[][] targets) throws Exception
   {
      PrintWriter pw = new PrintWriter(new FileWriter(filename));
      for (double[] row : targets)
      {
         StringBuilder sb = new StringBuilder();
         for (int i = 0; i < row.length; ++i)
         {
            if (i > 0) sb.append(" ");
            sb.append(row[i]);
         }
         pw.println(sb.toString());
      }
      pw.close();
      System.out.println("Saved truth table to: " + filename);
   }

   public static void main(String[] args) throws Exception
   {

      double[][] inputs2in4case =
      {
         {0.0, 0.0},
         {0.0, 1.0},
         {1.0, 0.0},
         {1.0, 1.0}
      };

      double[][] inputs3in8case =
      {
         {0.0, 0.0, 0.0},
         {0.0, 0.0, 1.0},
         {0.0, 1.0, 0.0},
         {0.0, 1.0, 1.0},
         {1.0, 0.0, 0.0},
         {1.0, 0.0, 1.0},
         {1.0, 1.0, 0.0},
         {1.0, 1.0, 1.0}
      };

      double[][] truth3out4case =
      {
         {0.0, 0.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 1.0},
         {1.0, 1.0, 0.0}
      };

      double[][] truthOr4case =
      {
         {0.0},
         {1.0},
         {1.0},
         {1.0}
      };


      double[][] truth3out8case =
      {
         {0.0, 0.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 0.0},
         {0.0, 1.0, 0.0},
         {1.0, 1.0, 1.0}
      };

      double[][] inputs2in12case =
      {
         { 0.1,  0.1}, { 0.1,  0.9}, { 0.9,  0.1}, { 0.9,  0.9},
         {-0.1, -0.1}, {-0.1,  1.1}, { 1.1, -0.1}, { 1.1,  1.1},
         {-0.1,  0.1}, {-0.1,  1.1}, { 0.9,  0.1}, { 0.9,  1.1}
      };

      double[][] truth12case3out =
      {
         {0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 0.0},
         {0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 0.0},
         {0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 0.0}
      };

      saveInputs("data/inputs/inputs_2in_4case.bin", inputs2in4case);
      saveInputs("data/inputs/inputs_3in_8case.bin", inputs3in8case);
      saveInputs("data/inputs/inputs_2in_12case.bin", inputs2in12case);
      saveTruthTable("data/truth/truth_andorxor_4case_3out.txt", truth3out4case);
      saveTruthTable("data/truth/truth_or_4case_1out.txt", truthOr4case);
      saveTruthTable("data/truth/truth_andorxor_8case_3out.txt", truth3out8case);
      saveTruthTable("data/truth/truth_12case_3out.txt", truth12case3out);
   }
}
