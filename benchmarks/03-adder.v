// Benchmark "3-adder" written by ABC on Tue May 23 15:44:41 2017

module 3-adder ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6,
    po0, po1, po2, po3 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6;
  output po0, po1, po2, po3;
  wire n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n24, n25, n26,
    n27, n28, n29, n30, n32, n33, n34, n36, n37, n38;
  assign n12 = ~pi0 & ~pi2;
  assign n13 = pi0 & pi2;
  assign n14 = pi1 & pi3;
  assign n15 = ~pi1 & ~pi3;
  assign n16 = ~pi5 & ~pi6;
  assign n17 = pi4 & ~n16;
  assign n18 = pi5 & pi6;
  assign n19 = ~n17 & ~n18;
  assign n20 = ~n15 & ~n19;
  assign n21 = ~n14 & ~n20;
  assign n22 = ~n13 & n21;
  assign po0 = ~n12 & ~n22;
  assign n24 = ~n12 & ~n13;
  assign n25 = ~n21 & ~n24;
  assign n26 = ~pi4 & ~n18;
  assign n27 = ~n16 & ~n26;
  assign n28 = ~n14 & ~n27;
  assign n29 = ~n15 & ~n28;
  assign n30 = n24 & ~n29;
  assign po1 = n25 | n30;
  assign n32 = ~n14 & ~n15;
  assign n33 = ~n19 & ~n32;
  assign n34 = ~n27 & n32;
  assign po2 = n33 | n34;
  assign n36 = ~n16 & ~n18;
  assign n37 = pi4 & ~n36;
  assign n38 = ~pi4 & n36;
  assign po3 = n37 | n38;
endmodule


