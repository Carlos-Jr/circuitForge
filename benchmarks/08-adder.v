// Benchmark "8-adder" written by ABC on Tue May 23 15:44:42 2017

module 8-adder ( 
    pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09, pi10, pi11,
    pi12, pi13, pi14, pi15, pi16,
    po0, po1, po2, po3, po4, po5, po6, po7, po8 );
  input  pi00, pi01, pi02, pi03, pi04, pi05, pi06, pi07, pi08, pi09,
    pi10, pi11, pi12, pi13, pi14, pi15, pi16;
  output po0, po1, po2, po3, po4, po5, po6, po7, po8;
  wire n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37, n38, n39, n40,
    n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51, n52, n53, n54,
    n55, n56, n57, n58, n59, n60, n61, n63, n64, n65, n66, n67, n68, n69,
    n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83,
    n84, n85, n86, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98,
    n99, n100, n102, n103, n104, n105, n106, n108, n109, n110, n111, n112,
    n113, n114, n116, n117, n118, n120, n121, n122, n123, n124, n125, n127,
    n128, n129, n131, n132, n133;
  assign n27 = ~pi00 & ~pi07;
  assign n28 = ~pi01 & ~pi08;
  assign n29 = pi02 & pi09;
  assign n30 = pi01 & pi08;
  assign n31 = ~n29 & ~n30;
  assign n32 = ~n28 & ~n31;
  assign n33 = ~n27 & n32;
  assign n34 = pi00 & pi07;
  assign n35 = ~pi04 & ~pi11;
  assign n36 = ~pi03 & ~pi10;
  assign n37 = ~n35 & ~n36;
  assign n38 = ~pi05 & ~pi12;
  assign n39 = ~pi15 & ~pi16;
  assign n40 = pi14 & ~n39;
  assign n41 = pi15 & pi16;
  assign n42 = ~n40 & ~n41;
  assign n43 = ~pi06 & ~pi13;
  assign n44 = ~n42 & ~n43;
  assign n45 = ~n38 & n44;
  assign n46 = pi06 & pi13;
  assign n47 = pi05 & pi12;
  assign n48 = ~n46 & ~n47;
  assign n49 = ~n38 & ~n48;
  assign n50 = ~n45 & ~n49;
  assign n51 = n37 & ~n50;
  assign n52 = pi04 & pi11;
  assign n53 = pi03 & pi10;
  assign n54 = ~n52 & ~n53;
  assign n55 = ~n36 & ~n54;
  assign n56 = ~n51 & ~n55;
  assign n57 = ~pi02 & ~pi09;
  assign n58 = ~n28 & ~n57;
  assign n59 = ~n27 & n58;
  assign n60 = ~n56 & n59;
  assign n61 = ~n33 & ~n34;
  assign po0 = n60 | ~n61;
  assign n63 = ~n27 & ~n34;
  assign n64 = ~pi14 & ~n41;
  assign n65 = ~n39 & ~n64;
  assign n66 = n48 & ~n65;
  assign n67 = ~n38 & ~n43;
  assign n68 = ~n47 & ~n67;
  assign n69 = ~n66 & ~n68;
  assign n70 = n31 & n54;
  assign n71 = ~n69 & n70;
  assign n72 = n63 & ~n71;
  assign n73 = n37 & n58;
  assign n74 = n49 & n73;
  assign n75 = n55 & n58;
  assign n76 = n45 & n73;
  assign n77 = ~n32 & ~n63;
  assign n78 = ~n74 & n77;
  assign n79 = ~n75 & n78;
  assign n80 = ~n76 & n79;
  assign n81 = ~n72 & ~n80;
  assign n82 = ~n30 & ~n58;
  assign n83 = ~n37 & ~n53;
  assign n84 = n31 & n83;
  assign n85 = ~n82 & ~n84;
  assign n86 = n63 & ~n85;
  assign po1 = n81 | n86;
  assign n88 = ~n28 & ~n30;
  assign n89 = n55 & ~n57;
  assign n90 = n37 & ~n57;
  assign n91 = ~n50 & n90;
  assign n92 = ~n29 & ~n89;
  assign n93 = ~n91 & n92;
  assign n94 = ~n88 & ~n93;
  assign n95 = ~n29 & n83;
  assign n96 = ~n57 & ~n95;
  assign n97 = n88 & ~n96;
  assign n98 = ~n28 & n70;
  assign n99 = ~n69 & n98;
  assign n100 = ~n97 & ~n99;
  assign po2 = n94 | ~n100;
  assign n102 = ~n29 & ~n57;
  assign n103 = ~n56 & ~n102;
  assign n104 = n54 & ~n69;
  assign n105 = ~n83 & ~n104;
  assign n106 = n102 & ~n105;
  assign po3 = n103 | n106;
  assign n108 = ~n36 & ~n53;
  assign n109 = ~n35 & ~n50;
  assign n110 = ~n52 & ~n109;
  assign n111 = ~n108 & ~n110;
  assign n112 = ~n52 & ~n69;
  assign n113 = ~n35 & ~n112;
  assign n114 = n108 & ~n113;
  assign po4 = n111 | n114;
  assign n116 = ~n35 & ~n52;
  assign n117 = ~n50 & ~n116;
  assign n118 = ~n69 & n116;
  assign po5 = n117 | n118;
  assign n120 = ~n38 & ~n47;
  assign n121 = ~n44 & ~n46;
  assign n122 = ~n120 & ~n121;
  assign n123 = ~n46 & ~n65;
  assign n124 = ~n43 & ~n123;
  assign n125 = n120 & ~n124;
  assign po6 = n122 | n125;
  assign n127 = ~n43 & ~n46;
  assign n128 = ~n42 & ~n127;
  assign n129 = ~n65 & n127;
  assign po7 = n128 | n129;
  assign n131 = ~n39 & ~n41;
  assign n132 = pi14 & ~n131;
  assign n133 = ~pi14 & n131;
  assign po8 = n132 | n133;
endmodule


