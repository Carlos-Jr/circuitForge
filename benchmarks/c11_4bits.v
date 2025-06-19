// Benchmark "top" written by ABC on Tue Apr  9 21:51:16 2024

module top ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8,
    po0, po1, po2, po3, po4  );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8;
  output po0, po1, po2, po3, po4;
  wire n10, n11, n12, n13, n14, n16,
    n17, n18, n19, n20, n21, n22, n23,
    n25, n26, n27, n28, n29, n30, n31, n32,
    n34, n35, n36, n37, n38, n39, n40,
    n41, n43, n44;
  assign n10 = ~pi0 & pi4;
  assign n11 = pi0 & ~pi4;
  assign n12 = ~n10 & ~n11;
  assign n13 = pi8 & n12;
  assign n14 = ~pi8 & ~n12;
  assign po0 = n13 | n14;
  assign n16 = ~pi1 & pi5;
  assign n17 = pi1 & ~pi5;
  assign n18 = ~n16 & ~n17;
  assign n19 = pi8 & ~n12;
  assign n20 = pi0 & pi4;
  assign n21 = ~n19 & ~n20;
  assign n22 = n18 & ~n21;
  assign n23 = ~n18 & n21;
  assign po1 = n22 | n23;
  assign n25 = ~pi2 & pi6;
  assign n26 = pi2 & ~pi6;
  assign n27 = ~n25 & ~n26;
  assign n28 = ~n18 & ~n21;
  assign n29 = pi1 & pi5;
  assign n30 = ~n28 & ~n29;
  assign n31 = n27 & ~n30;
  assign n32 = ~n27 & n30;
  assign po2 = n31 | n32;
  assign n34 = ~pi3 & pi7;
  assign n35 = pi3 & ~pi7;
  assign n36 = ~n34 & ~n35;
  assign n37 = ~n27 & ~n30;
  assign n38 = pi2 & pi6;
  assign n39 = ~n37 & ~n38;
  assign n40 = n36 & ~n39;
  assign n41 = ~n36 & n39;
  assign po3 = n40 | n41;
  assign n43 = ~n36 & ~n39;
  assign n44 = pi3 & pi7;
  assign po4 = n43 | n44;
endmodule


