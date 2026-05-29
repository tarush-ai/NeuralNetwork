/*
* Author: Tarush Gupta
* Date: May 25, 2026
* Description: Standalone image preprocessing tool for the finger-counting
*              training pipeline. Reads a raw grayscale byte file (output of
*              BMP2OneByte), wraps it in a PelArray, optionally rotates 180
*              (when the source BMPs are upside down), takes the ones complement
*              so the hand becomes the "heavy" mass over a 0 background, finds
*              the center of mass, crops a configurable rectangular source window
*              around it, scales that window to the target output dimensions,
*              and writes the cropped grayscale bytes in BMP bottom-up storage
*              order so BGR2BMP can render the result without further adjustment.
*
*              This tool is part of preprocessing only. It does NOT touch
*              NLayer.java or any network code.
*
* Usage:
*    javac Normalize.java
*    java Normalize inWidth inHeight outWidth outHeight inFile outFile [flags]
*
* Flags:
*    --no-rotate         skip the 180-degree image-content rotation
*    --no-complement     skip the ones complement (use when source already has black bg)
*    --no-output-flip    skip the bottom-up storage flip on output (default is to flip)
*    --com-threshold N   zero out any complemented gray value below N (0-255) for COM
*                        purposes only. The crop itself uses the pre-threshold image.
*                        Default 0 (no thresholding). Use ~150 when the source has
*                        gray (not pure white) background.
*    --margins T R B L   zero out the top T rows, right R cols, bottom B rows, left L cols
*                        for COM purposes only. Use to ignore dark camera-edge artifacts.
*    --scale-from W H    crop a W-by-H window around COM, then scale to outWidth-by-outHeight.
*                        Default outWidth-by-outHeight (no scaling). Use a larger W/H when
*                        the hand spans more pels than the output dimensions in the original.
*
* Example:
*    java Normalize 4032 3024 277 256 data/raw_bytes/1_1.bin data/norm_bytes/1_1.bin \
*       --no-rotate --com-threshold 180 --margins 300 1500 100 100 --scale-from 1300 1200
*/
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class Normalize
{
/*
* Entry point. Parses dims, in/out paths, and optional flags, then runs the pipeline.
*/
   public static void main(String[] args) throws Exception
   {
      int width;
      int height;
      int outWidth;
      int outHeight;
      String inFile;
      String outFile;
      boolean doRotate;
      boolean doComplement;
      boolean outputBottomUp;
      int comThreshold;
      int marginTop;
      int marginRight;
      int marginBottom;
      int marginLeft;
      int scaleFromW;
      int scaleFromH;
      int i;

      if (args.length < 6)
      {
         System.out.println("Usage: java Normalize inWidth inHeight outWidth outHeight inFile outFile " +
                            "[--no-rotate] [--no-complement] [--no-output-flip] " +
                            "[--com-threshold N] [--margins T R B L] [--scale-from W H]");
         return;
      }

      width     = Integer.parseInt(args[0]);
      height    = Integer.parseInt(args[1]);
      outWidth  = Integer.parseInt(args[2]);
      outHeight = Integer.parseInt(args[3]);
      inFile    = args[4];
      outFile   = args[5];

      doRotate       = true;
      doComplement   = true;
      outputBottomUp = true;
      comThreshold   = 0;
      marginTop      = 0;
      marginRight    = 0;
      marginBottom   = 0;
      marginLeft     = 0;
      scaleFromW     = outWidth;
      scaleFromH     = outHeight;

      for (i = 6; i < args.length; ++i)
      {
         if (args[i].equals("--no-rotate"))      doRotate = false;
         if (args[i].equals("--no-complement"))  doComplement = false;
         if (args[i].equals("--no-output-flip")) outputBottomUp = false;
         if (args[i].equals("--com-threshold") && i + 1 < args.length)
         {
            comThreshold = Integer.parseInt(args[i + 1]);
            ++i;
         }
         if (args[i].equals("--margins") && i + 4 < args.length)
         {
            marginTop    = Integer.parseInt(args[i + 1]);
            marginRight  = Integer.parseInt(args[i + 2]);
            marginBottom = Integer.parseInt(args[i + 3]);
            marginLeft   = Integer.parseInt(args[i + 4]);
            i += 4;
         }
         if (args[i].equals("--scale-from") && i + 2 < args.length)
         {
            scaleFromW = Integer.parseInt(args[i + 1]);
            scaleFromH = Integer.parseInt(args[i + 2]);
            i += 2;
         }
      }

/*
* BMP2OneByte writes bytes in bottom-up storage order: row (height-1) first, row 0 last.
* Read into a top-down image-space int[][] so PelArray operations (COM, crop) work in
* natural image coordinates.
*/
      int[][] pels = new int[height][width];
      DataInputStream in = new DataInputStream(new FileInputStream(inFile));
      int row;
      int col;
      int g;

      for (row = height - 1; row >= 0; --row)
      {
         for (col = 0; col < width; ++col)
         {
            g = in.readUnsignedByte();
            pels[row][col] = (g << 16) | (g << 8) | g;
         }
      }
      in.close();

      PelArray img = new PelArray(pels);

/*
* The source iPhone JPGs carry "lower-right" EXIF orientation, which means the pixel
* content is rotated 180 degrees relative to a normal upright photo. Rotate twice
* clockwise to put the hand right-side-up before COM is computed.
*/
      if (doRotate)
      {
         img = img.rotateCW90().rotateCW90();
      }

/*
* White background images have high pel values everywhere except the hand. COM in
* PelArray weights pels by their value, so without complementing the COM lands in
* the middle of the white background. Complementing makes background 0 and the hand
* the heaviest region.
*/
      if (doComplement)
      {
         img = img.onesComplimentImage();
      }

/*
* When the source background isn't pure white, even the complemented background carries
* enough mass to drag the COM off the hand. Build a separate thresholded copy for the
* COM computation only — the crop itself still uses the original (complemented) pels so
* no detail is destroyed in the final activations.
*/
      PelArray comSource = img;
      if (comThreshold > 0 || marginTop > 0 || marginRight > 0 || marginBottom > 0 || marginLeft > 0)
      {
         int[][] src = img.getPelArray();
         int h = src.length;
         int w = src[0].length;
         int[][] thr = new int[h][w];
         int r;
         int c;
         int v;
         for (r = 0; r < h; ++r)
         {
            for (c = 0; c < w; ++c)
            {
               v = src[r][c] & 0xFF;
               if (v < comThreshold) v = 0;
               if (r < marginTop || r >= h - marginBottom) v = 0;
               if (c < marginLeft || c >= w - marginRight) v = 0;
               thr[r][c] = (v << 16) | (v << 8) | v;
            }
         }
         comSource = new PelArray(thr);
      }

/*
* Find COM and place a scaleFromW-by-scaleFromH window centered on it. Clamp the
* window to image bounds so the crop never reads out of range.
*/
      comSource.calcCOM();
      int xCom = comSource.getXcom();
      int yCom = comSource.getYcom();
      int imgW = img.getWidth();
      int imgH = img.getHeight();

      int halfW = scaleFromW / 2;
      int halfH = scaleFromH / 2;
      int xUL = xCom - halfW;
      int yUL = yCom - halfH;
      int xLR = xUL + scaleFromW - 1;
      int yLR = yUL + scaleFromH - 1;

      if (xUL < 0)     { xUL = 0;        xLR = scaleFromW - 1; }
      if (yUL < 0)     { yUL = 0;        yLR = scaleFromH - 1; }
      if (xLR >= imgW) { xLR = imgW - 1; xUL = xLR - scaleFromW + 1; }
      if (yLR >= imgH) { yLR = imgH - 1; yUL = yLR - scaleFromH + 1; }

      System.out.printf("COM=(%d,%d) crop=(%d,%d)-(%d,%d) src=%dx%d out=%dx%d%n",
                        xCom, yCom, xUL, yUL, xLR, yLR,
                        scaleFromW, scaleFromH, outWidth, outHeight);

      PelArray cropped = img.crop(xUL, yUL, xLR, yLR);
      if (scaleFromW != outWidth || scaleFromH != outHeight)
      {
         cropped = cropped.scale(outWidth, outHeight);
      }
      int[][] outPels = cropped.getPelArray();

/*
* Write the cropped image. Output order matches BMP bottom-up storage so BGR2BMP
* renders the image upright. Each pel is grayscale (R=G=B); take the low byte.
*/
      DataOutputStream out = new DataOutputStream(new FileOutputStream(outFile));

      if (outputBottomUp)
      {
         for (row = outHeight - 1; row >= 0; --row)
         {
            for (col = 0; col < outWidth; ++col)
            {
               out.writeByte((byte) (outPels[row][col] & 0xFF));
            }
         }
      }
      else
      {
         for (row = 0; row < outHeight; ++row)
         {
            for (col = 0; col < outWidth; ++col)
            {
               out.writeByte((byte) (outPels[row][col] & 0xFF));
            }
         }
      }
      out.close();
   } // main(String[] args)
} // class Normalize
