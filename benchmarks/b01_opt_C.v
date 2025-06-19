// Benchmark "b01_opt_C" written by ABC on Thu Oct  5 16:36:41 2017

module b01_opt_C (
    pi0, pi1, pi2, pi3, pi4, pi5, pi6,
    po0, po1, po2, po3, po4, po5, po6 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6;
  output po0, po1, po2, po3, po4, po5, po6;
  wire n15, n16, n17, n18, n19, n20, n21, n22, n24, n25, n26, n27, n28, n29,
    n30, n31, n32, n34, n35, n36, n37, n38, n39, n40, n41, n43, n44, n45,
    n47, n48, n49, n50, n51, n52;
  assign n15 = ~pi0 & ~pi1;
  assign n16 = ~pi5 & n15;
  assign n17 = ~pi4 & ~n16;
  assign n18 = pi3 & n17;
  assign n19 = pi4 & ~pi5;
  assign n20 = pi0 & pi1;
  assign n21 = ~n19 & ~n20;
  assign n22 = ~pi3 & ~n21;
  assign po2 = n18 | n22;
  assign n24 = pi3 & ~n20;
  assign n25 = n19 & ~n24;
  assign n26 = pi3 & ~n15;
  assign n27 = pi5 & n26;
  assign n28 = pi5 & ~n20;
  assign n29 = pi3 & n15;
  assign n30 = ~n28 & ~n29;
  assign n31 = ~pi4 & ~n30;
  assign n32 = ~n25 & ~n27;
  assign po3 = n31 | ~n32;
  assign n34 = pi3 & ~n19;
  assign n35 = ~n15 & n34;
  assign n36 = ~pi5 & n20;
  assign n37 = ~pi3 & ~pi4;
  assign n38 = ~n36 & n37;
  assign n39 = ~n28 & n38;
  assign n40 = n19 & n20;
  assign n41 = ~pi3 & pi4;
  assign po6 = pi5 & n41;
  assign n43 = ~n20 & po6;
  assign n44 = ~n35 & ~n39;
  assign n45 = ~n40 & n44;
  assign po4 = n43 | ~n45;
  assign n47 = ~pi0 & pi1;
  assign n48 = pi0 & ~pi1;
  assign n49 = ~n47 & ~n48;
  assign n50 = ~n34 & ~n49;
  assign n51 = ~n15 & ~n20;
  assign n52 = n34 & ~n51;
  assign po5 = n50 | n52;
  assign po0 = pi6;
  assign po1 = pi2;
endmodule
