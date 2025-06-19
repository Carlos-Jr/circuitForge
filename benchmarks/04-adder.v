// Benchmark "4-adder" written by ABC on Tue May 23 15:44:42 2017

module 4-adder ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8,
    po0, po1, po2, po3, po4 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8;
  output po0, po1, po2, po3, po4;
  wire n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28,
    n29, n30, n32, n33, n34, n35, n36, n37, n38, n39, n41, n42, n43, n44,
    n45, n46, n47, n49, n50, n51, n53, n54, n55;
  assign n15 = pi0 & pi3;
  assign n16 = ~pi0 & ~pi3;
  assign n17 = ~pi7 & ~pi8;
  assign n18 = pi6 & ~n17;
  assign n19 = pi7 & pi8;
  assign n20 = ~n18 & ~n19;
  assign n21 = ~pi1 & ~pi4;
  assign n22 = ~pi2 & ~pi5;
  assign n23 = ~n21 & ~n22;
  assign n24 = ~n20 & n23;
  assign n25 = pi2 & pi5;
  assign n26 = pi1 & pi4;
  assign n27 = ~n25 & ~n26;
  assign n28 = ~n21 & ~n27;
  assign n29 = ~n24 & ~n28;
  assign n30 = ~n16 & ~n29;
  assign po0 = n15 | n30;
  assign n32 = ~n15 & ~n16;
  assign n33 = ~n29 & ~n32;
  assign n34 = ~n23 & ~n26;
  assign n35 = ~pi6 & ~n19;
  assign n36 = ~n17 & ~n35;
  assign n37 = n27 & ~n36;
  assign n38 = ~n34 & ~n37;
  assign n39 = n32 & ~n38;
  assign po1 = n33 | n39;
  assign n41 = ~n20 & ~n22;
  assign n42 = ~n25 & ~n41;
  assign n43 = ~n21 & ~n26;
  assign n44 = ~n42 & ~n43;
  assign n45 = ~n25 & ~n36;
  assign n46 = ~n22 & ~n45;
  assign n47 = n43 & ~n46;
  assign po2 = n44 | n47;
  assign n49 = ~n22 & ~n25;
  assign n50 = ~n20 & ~n49;
  assign n51 = ~n36 & n49;
  assign po3 = n50 | n51;
  assign n53 = ~n17 & ~n19;
  assign n54 = pi6 & ~n53;
  assign n55 = ~pi6 & n53;
  assign po4 = n54 | n55;
endmodule


