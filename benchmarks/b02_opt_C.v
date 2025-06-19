// Benchmark "b02_opt_C" written by ABC on Thu Oct  5 16:36:41 2017

module b02_opt_C ( 
    pi0, pi1, pi2, pi3, pi4,
    po0, po1, po2, po3, po4 );
  input  pi0, pi1, pi2, pi3, pi4;
  output po0, po1, po2, po3, po4;
  wire n11, n12, n13, n14, n15, n16, n17, n19, n20, n21, n22, n23, n25, n26,
    n27, n28, n29, n31;
  assign n11 = ~pi1 & ~pi2;
  assign n12 = ~pi3 & n11;
  assign n13 = pi4 & ~n12;
  assign n14 = ~pi1 & pi2;
  assign n15 = pi1 & ~pi2;
  assign n16 = ~n14 & ~n15;
  assign n17 = pi3 & ~n16;
  assign po1 = n13 | n17;
  assign n19 = ~pi2 & ~pi4;
  assign n20 = pi3 & n19;
  assign n21 = ~pi1 & ~pi3;
  assign n22 = ~pi2 & ~n21;
  assign n23 = pi4 & ~n22;
  assign po2 = n20 | n23;
  assign n25 = ~pi3 & ~pi4;
  assign n26 = pi1 & pi3;
  assign n27 = ~pi1 & pi4;
  assign n28 = ~n26 & ~n27;
  assign n29 = ~pi2 & n28;
  assign po3 = n25 | n29;
  assign n31 = pi2 & ~pi3;
  assign po4 = ~pi4 & n31;
  assign po0 = pi0;
endmodule


