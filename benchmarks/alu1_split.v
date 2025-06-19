// Benchmark "alu1_split" written by ABC on Tue Nov 21 00:27:51 2017

module alu1_split ( 
    pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09, pi10, pi11,
    po0, po1, po2, po3, po4, po5, po6, po7 );
  input  pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09,
    pi10, pi11;
  output po0, po1, po2, po3, po4, po5, po6, po7;
  wire n21, n22, n23, n24, n26, n27, n28, n29, n31, n32, n33, n34, n36, n37,
    n38, n39, n41, n42, n43, n45, n46, n47, n49, n50, n51, n53;
  assign n21 = ~pi04 & ~pi10;
  assign n22 = pi04 & ~pi11;
  assign n23 = ~n21 & ~n22;
  assign n24 = pi00 & ~n23;
  assign po0 = ~pi00 | n24;
  assign n26 = ~pi05 & ~pi10;
  assign n27 = pi05 & ~pi11;
  assign n28 = ~n26 & ~n27;
  assign n29 = pi01 & ~n28;
  assign po1 = ~pi01 | n29;
  assign n31 = ~pi06 & ~pi10;
  assign n32 = pi06 & ~pi11;
  assign n33 = ~n31 & ~n32;
  assign n34 = pi02 & ~n33;
  assign po2 = ~pi02 | n34;
  assign n36 = ~pi07 & ~pi10;
  assign n37 = pi07 & ~pi11;
  assign n38 = ~n36 & ~n37;
  assign n39 = pi03 & ~n38;
  assign po3 = ~pi03 | n39;
  assign n41 = pi04 & ~pi08;
  assign n42 = ~pi04 & ~pi09;
  assign n43 = ~n41 & ~n42;
  assign po4 = ~pi00 & ~n43;
  assign n45 = pi05 & ~pi08;
  assign n46 = ~pi05 & ~pi09;
  assign n47 = ~n45 & ~n46;
  assign po5 = ~pi01 & ~n47;
  assign n49 = pi06 & ~pi08;
  assign n50 = ~pi06 & ~pi09;
  assign n51 = ~n49 & ~n50;
  assign po6 = ~pi02 & ~n51;
  assign n53 = ~pi03 & pi07;
  assign po7 = ~pi08 & n53;
endmodule


