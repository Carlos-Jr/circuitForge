// Benchmark "5-adder" written by ABC on Tue May 23 15:44:42 2017

module 5-adder (
    pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09, pi10,
    po0, po1, po2, po3, po4, po5 );
  input  pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09,
    pi10;
  output po0, po1, po2, po3, po4, po5;
  wire n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31,
    n32, n33, n34, n35, n36, n37, n39, n40, n41, n42, n43, n44, n45, n46,
    n47, n48, n49, n50, n51, n53, n54, n55, n56, n57, n59, n60, n61, n62,
    n63, n64, n66, n67, n68, n70, n71, n72;
  assign n18 = pi00 & pi04;
  assign n19 = ~pi01 & ~pi05;
  assign n20 = ~pi00 & ~pi04;
  assign n21 = pi01 & pi05;
  assign n22 = ~pi02 & ~pi06;
  assign n23 = pi03 & pi07;
  assign n24 = pi02 & pi06;
  assign n25 = ~n23 & ~n24;
  assign n26 = ~n22 & ~n25;
  assign n27 = ~pi09 & ~pi10;
  assign n28 = pi08 & ~n27;
  assign n29 = pi09 & pi10;
  assign n30 = ~n28 & ~n29;
  assign n31 = ~pi03 & ~pi07;
  assign n32 = ~n30 & ~n31;
  assign n33 = ~n22 & n32;
  assign n34 = ~n21 & ~n26;
  assign n35 = ~n33 & n34;
  assign n36 = ~n19 & ~n20;
  assign n37 = ~n35 & n36;
  assign po0 = n18 | n37;
  assign n39 = ~n18 & ~n20;
  assign n40 = ~n19 & ~n39;
  assign n41 = ~n35 & n40;
  assign n42 = ~n22 & ~n31;
  assign n43 = ~n21 & ~n24;
  assign n44 = ~n42 & n43;
  assign n45 = ~pi08 & ~n29;
  assign n46 = ~n27 & ~n45;
  assign n47 = ~n21 & n25;
  assign n48 = ~n46 & n47;
  assign n49 = ~n19 & ~n44;
  assign n50 = ~n48 & n49;
  assign n51 = n39 & ~n50;
  assign po1 = n41 | n51;
  assign n53 = ~n19 & ~n21;
  assign n54 = ~n26 & ~n33;
  assign n55 = ~n53 & ~n54;
  assign n56 = ~n44 & ~n48;
  assign n57 = ~n19 & ~n56;
  assign po2 = n55 | n57;
  assign n59 = ~n22 & ~n24;
  assign n60 = ~n23 & ~n32;
  assign n61 = ~n59 & ~n60;
  assign n62 = ~n23 & ~n46;
  assign n63 = ~n31 & ~n62;
  assign n64 = n59 & ~n63;
  assign po3 = n61 | n64;
  assign n66 = ~n23 & ~n31;
  assign n67 = ~n30 & ~n66;
  assign n68 = ~n46 & n66;
  assign po4 = n67 | n68;
  assign n70 = ~n27 & ~n29;
  assign n71 = pi08 & ~n70;
  assign n72 = ~pi08 & n70;
  assign po5 = n71 | n72;
endmodule
