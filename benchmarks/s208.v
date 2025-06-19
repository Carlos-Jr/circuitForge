// Benchmark "s208.blif" written by ABC on Fri Dec  8 15:24:49 2017

module s208.blif ( 
    pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09, pi10, pi11,
    pi12, pi13, pi14, pi15, pi16, pi17, pi18,
    po0, po1, po2, po3, po4, po5, po6, po7, po8, po9 );
  input  pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09,
    pi10, pi11, pi12, pi13, pi14, pi15, pi16, pi17, pi18;
  output po0, po1, po2, po3, po4, po5, po6, po7, po8, po9;
  wire n30, n32, n33, n34, n35, n36, n37, n38, n39, n41, n42, n43, n45, n46,
    n48, n49, n50, n51, n52, n53, n54, n55, n57, n58, n59, n61, n62, n63,
    n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77,
    n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91,
    n93, n94, n96, n97;
  assign n30 = pi00 & ~pi01;
  assign po0 = ~pi14 & n30;
  assign n32 = pi14 & n30;
  assign n33 = pi13 & n32;
  assign n34 = ~pi11 & pi12;
  assign n35 = n33 & n34;
  assign n36 = pi13 & pi14;
  assign n37 = pi12 & n36;
  assign n38 = n30 & ~n37;
  assign n39 = pi11 & n38;
  assign po1 = n35 | n39;
  assign n41 = n30 & ~n36;
  assign n42 = pi12 & n41;
  assign n43 = ~pi12 & n33;
  assign po2 = n42 | n43;
  assign n45 = pi11 & n37;
  assign n46 = ~pi01 & n45;
  assign po3 = ~pi18 & n46;
  assign n48 = pi18 & n46;
  assign n49 = pi17 & n48;
  assign n50 = ~pi15 & pi16;
  assign n51 = n49 & n50;
  assign n52 = pi17 & pi18;
  assign n53 = pi16 & n52;
  assign n54 = n46 & ~n53;
  assign n55 = pi15 & n54;
  assign po4 = n51 | n55;
  assign n57 = n46 & ~n52;
  assign n58 = pi16 & n57;
  assign n59 = ~pi16 & n49;
  assign po5 = n58 | n59;
  assign n61 = pi00 & ~pi14;
  assign n62 = ~pi13 & n61;
  assign n63 = ~pi12 & n62;
  assign n64 = ~pi11 & n63;
  assign n65 = pi18 & n64;
  assign n66 = pi05 & n65;
  assign n67 = pi13 & n61;
  assign n68 = pi08 & n67;
  assign n69 = ~n66 & ~n68;
  assign n70 = ~pi18 & n64;
  assign n71 = pi17 & n70;
  assign n72 = pi04 & n71;
  assign n73 = ~pi17 & n70;
  assign n74 = pi16 & n73;
  assign n75 = pi03 & n74;
  assign n76 = pi00 & pi10;
  assign n77 = pi00 & pi14;
  assign n78 = pi09 & n77;
  assign n79 = ~n76 & ~n78;
  assign n80 = ~n72 & ~n75;
  assign n81 = n79 & n80;
  assign n82 = n69 & n81;
  assign n83 = pi12 & n62;
  assign n84 = pi07 & n83;
  assign n85 = pi11 & n63;
  assign n86 = pi06 & n85;
  assign n87 = ~n84 & ~n86;
  assign n88 = ~pi16 & n73;
  assign n89 = pi15 & n88;
  assign n90 = pi02 & n89;
  assign n91 = n82 & n87;
  assign po6 = n90 | ~n91;
  assign n93 = pi13 & po0;
  assign n94 = ~pi13 & n32;
  assign po7 = n93 | n94;
  assign n96 = pi17 & po3;
  assign n97 = ~pi17 & n48;
  assign po8 = n96 | n97;
  assign po9 = pi15 & n53;
endmodule


