// Benchmark "b02_C" written by ABC on Thu Oct  5 16:36:41 2017

module b02_C ( 
    pi0, pi1, pi2, pi3, pi4,
    po0, po1, po2, po3, po4 );
  input  pi0, pi1, pi2, pi3, pi4;
  output po0, po1, po2, po3, po4;
  wire n11, n12, n13, n14, n15, n16, n17, n19, n20, n21, n22, n23, n25, n26,
    n27, n29;
  assign n11 = ~pi1 & pi2;
  assign n12 = pi1 & ~pi2;
  assign n13 = ~n11 & ~n12;
  assign n14 = ~pi4 & n13;
  assign n15 = pi3 & ~n14;
  assign n16 = ~pi1 & ~pi2;
  assign n17 = pi4 & ~n16;
  assign po1 = n15 | n17;
  assign n19 = ~pi2 & ~pi4;
  assign n20 = pi3 & n19;
  assign n21 = ~pi1 & ~pi3;
  assign n22 = ~pi2 & ~n21;
  assign n23 = pi4 & ~n22;
  assign po2 = n20 | n23;
  assign n25 = pi4 & ~n12;
  assign n26 = ~pi3 & ~n25;
  assign n27 = ~pi4 & n16;
  assign po3 = n26 | n27;
  assign n29 = ~pi3 & ~pi4;
  assign po4 = pi2 & n29;
  assign po0 = pi0;
endmodule


