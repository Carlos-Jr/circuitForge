// Benchmark "6-adder" written by ABC on Tue May 23 15:44:42 2017

module 6-adder ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8, pi9, pi10, pi11,
pi12,
    po0, po1, po2, po3, po4, po5, po6 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8, pi9, pi10, pi11, pi12;
  output po0, po1, po2, po3, po4, po5, po6;
  wire n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34,
    n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n48, n49,
    n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, n60, n62, n63, n64,
    n65, n66, n67, n68, n70, n71, n72, n74, n75, n76, n77, n78, n79, n80,
    n82, n83, n84, n86, n87, n88;
  assign n21 = ~pi0 & ~pi5;
  assign n22 = ~pi1 & ~pi6;
  assign n23 = pi2 & pi7;
  assign n24 = pi1 & pi6;
  assign n25 = ~n23 & ~n24;
  assign n26 = ~n22 & ~n25;
  assign n27 = ~n21 & n26;
  assign n28 = pi0 & pi5;
  assign n29 = ~pi11 & ~pi12;
  assign n30 = pi10 & ~n29;
  assign n31 = pi11 & pi12;
  assign n32 = ~n30 & ~n31;
  assign n33 = ~pi4 & ~pi9;
  assign n34 = ~pi3 & ~pi8;
  assign n35 = ~n33 & ~n34;
  assign n36 = ~n32 & n35;
  assign n37 = pi4 & pi9;
  assign n38 = pi3 & pi8;
  assign n39 = ~n37 & ~n38;
  assign n40 = ~n34 & ~n39;
  assign n41 = ~n36 & ~n40;
  assign n42 = ~pi2 & ~pi7;
  assign n43 = ~n22 & ~n42;
  assign n44 = ~n21 & n43;
  assign n45 = ~n41 & n44;
  assign n46 = ~n27 & ~n28;
  assign po0 = n45 | ~n46;
  assign n48 = ~n21 & ~n28;
  assign n49 = ~n41 & n43;
  assign n50 = ~n26 & ~n49;
  assign n51 = ~n48 & ~n50;
  assign n52 = ~n24 & ~n43;
  assign n53 = ~pi10 & ~n31;
  assign n54 = ~n29 & ~n53;
  assign n55 = n39 & ~n54;
  assign n56 = ~n35 & ~n38;
  assign n57 = ~n55 & ~n56;
  assign n58 = n25 & ~n57;
  assign n59 = ~n52 & ~n58;
  assign n60 = n48 & ~n59;
  assign po1 = n51 | n60;
  assign n62 = ~n22 & ~n24;
  assign n63 = ~n41 & ~n42;
  assign n64 = ~n23 & ~n63;
  assign n65 = ~n62 & ~n64;
  assign n66 = ~n23 & ~n57;
  assign n67 = ~n42 & ~n66;
  assign n68 = n62 & ~n67;
  assign po2 = n65 | n68;
  assign n70 = ~n23 & ~n42;
  assign n71 = ~n41 & ~n70;
  assign n72 = ~n57 & n70;
  assign po3 = n71 | n72;
  assign n74 = ~n34 & ~n38;
  assign n75 = ~n32 & ~n33;
  assign n76 = ~n37 & ~n75;
  assign n77 = ~n74 & ~n76;
  assign n78 = ~n37 & ~n54;
  assign n79 = ~n33 & ~n78;
  assign n80 = n74 & ~n79;
  assign po4 = n77 | n80;
  assign n82 = ~n33 & ~n37;
  assign n83 = ~n32 & ~n82;
  assign n84 = ~n54 & n82;
  assign po5 = n83 | n84;
  assign n86 = ~n29 & ~n31;
  assign n87 = pi10 & ~n86;
  assign n88 = ~pi10 & n86;
  assign po6 = n87 | n88;
endmodule


