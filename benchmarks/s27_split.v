// Benchmark "/mnt/d/Dropbox/benchmarks/ISCAS/PLA/minterms/s27_split" written by ABC on Fri Dec  8 07:28:52 2017

module s27_split ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6,
    po0, po1, po2, po3 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6;
  output po0, po1, po2, po3;
  wire n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25,
    n26, n27, n28, n29, n30, n31, n33, n34, n35, n36, n37, n38, n40, n41,
    n42, n43, n44, n45, n46;
  assign n12 = pi1 & ~pi6;
  assign n13 = ~pi6 & ~n12;
  assign n14 = pi5 & ~n13;
  assign n15 = pi0 & n14;
  assign n16 = ~pi5 & ~pi6;
  assign n17 = pi1 & n16;
  assign n18 = ~n15 & ~n17;
  assign n19 = ~pi4 & ~n18;
  assign n20 = pi5 & pi6;
  assign n21 = pi6 & ~n20;
  assign n22 = pi4 & ~n21;
  assign n23 = ~n19 & ~n22;
  assign n24 = pi3 & ~n23;
  assign n25 = pi0 & ~pi4;
  assign n26 = ~pi4 & ~n25;
  assign n27 = pi5 & ~n26;
  assign n28 = ~n16 & ~n27;
  assign n29 = ~pi3 & ~n28;
  assign n30 = ~pi5 & pi6;
  assign n31 = ~n29 & ~n30;
  assign po0 = n24 | ~n31;
  assign n33 = pi1 & ~pi4;
  assign n34 = ~pi4 & ~n33;
  assign n35 = pi3 & ~n34;
  assign n36 = pi3 & ~n35;
  assign n37 = ~pi6 & ~n36;
  assign n38 = ~pi6 & ~n37;
  assign po1 = pi0 & ~n38;
  assign n40 = ~pi0 & n14;
  assign n41 = ~pi1 & ~pi6;
  assign n42 = ~n40 & ~n41;
  assign n43 = pi3 & ~n42;
  assign n44 = ~pi3 & pi5;
  assign n45 = ~pi0 & n44;
  assign n46 = ~n43 & ~n45;
  assign po2 = ~pi4 & ~n46;
  assign po3 = ~pi2 & ~n13;
endmodule
