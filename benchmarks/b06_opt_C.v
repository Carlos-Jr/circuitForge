// Benchmark "b06_opt_C" written by ABC on Thu Oct  5 16:36:42 2017

module b06_opt_C ( 
    pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09, pi10,
    po00, po01, po02, po03, po04, po05, po06, po07, po08, po09, po10, po11,
    po12, po13, po14 );
  input  pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09,
    pi10;
  output po00, po01, po02, po03, po04, po05, po06, po07, po08, po09, po10,
    po11, po12, po13, po14;
  wire n27, n28, n29, n30, n31, n32, n34, n35, n36, n37, n38, n39, n40, n41,
    n42, n44, n45, n46, n48, n49, n51, n52, n53, n54, n55, n56, n58, n60,
    n61, n63, n64, n65, n66;
  assign n27 = ~pi04 & ~pi05;
  assign n28 = ~pi00 & ~n27;
  assign n29 = pi03 & ~n28;
  assign n30 = ~pi03 & ~pi04;
  assign n31 = pi05 & n30;
  assign n32 = ~pi00 & n31;
  assign po06 = n29 | n32;
  assign n34 = ~pi03 & ~pi05;
  assign n35 = pi04 & n34;
  assign n36 = pi00 & ~pi03;
  assign n37 = pi05 & n36;
  assign n38 = pi00 & pi04;
  assign n39 = ~pi00 & n27;
  assign n40 = pi03 & n39;
  assign n41 = ~n35 & ~n37;
  assign n42 = ~n38 & n41;
  assign po07 = n40 | ~n42;
  assign n44 = pi04 & pi05;
  assign n45 = ~pi04 & n34;
  assign n46 = ~n44 & ~n45;
  assign po08 = n28 | ~n46;
  assign n48 = ~pi05 & n38;
  assign n49 = ~n29 & ~n31;
  assign po09 = n48 | ~n49;
  assign n51 = pi03 & pi05;
  assign n52 = pi00 & ~pi05;
  assign n53 = pi04 & ~n52;
  assign n54 = pi00 & ~pi04;
  assign n55 = ~n34 & ~n51;
  assign n56 = ~n53 & n55;
  assign po10 = n54 | ~n56;
  assign n58 = pi03 & n38;
  assign po11 = n40 | n58;
  assign n60 = pi03 & pi04;
  assign n61 = pi00 & ~n60;
  assign po12 = ~n46 | ~n61;
  assign n63 = pi05 & n60;
  assign n64 = ~pi02 & ~n63;
  assign n65 = ~pi00 & n34;
  assign n66 = pi04 & n65;
  assign po13 = n64 | n66;
  assign po00 = pi06;
  assign po01 = pi07;
  assign po02 = pi08;
  assign po03 = pi09;
  assign po04 = pi10;
  assign po05 = pi01;
  assign po14 = po13;
endmodule


