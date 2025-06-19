// Benchmark "top" written by ABC on Tue Oct 24 16:48:22 2017

module top ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7,
    po0, po1, po2, po3 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7;
  output po0, po1, po2, po3;
  wire n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26,
    n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37, n38, n39, n40,
    n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51, n52, n53, n54,
    n55, n56, n57, n58, n59, n60, n61, n62, n63, n64, n65, n66, n67, n68,
    n69, n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82,
    n83, n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96,
    n97, n98, n99, n100, n101, n102, n103, n104, n105, n106, n107, n108,
    n109, n110, n111, n112, n113, n114, n115, n116, n117, n118, n119, n120,
    n121, n122, n123, n124, n125, n126, n127, n128, n129, n130, n131, n132,
    n133, n134, n135, n136, n137, n138, n139, n140, n141, n142, n143, n144,
    n145, n146, n147, n148, n149, n150, n151, n152, n153, n154, n155, n156,
    n157, n158, n159, n160, n161, n162, n163, n164, n165, n166, n167, n168,
    n169, n170, n171, n172, n173, n174, n175, n176, n177, n178, n179, n180,
    n181, n182, n183, n185, n186, n188, n189, n191, n192;
  assign n13 = ~pi0 & ~pi1;
  assign n14 = pi0 & pi1;
  assign n15 = ~n13 & ~n14;
  assign n16 = pi1 & n13;
  assign n17 = ~n13 & ~n15;
  assign n18 = ~n16 & ~n17;
  assign n19 = pi3 & n13;
  assign n20 = pi3 & ~n13;
  assign n21 = ~n19 & ~n20;
  assign n22 = pi2 & n13;
  assign n23 = pi2 & ~n13;
  assign n24 = ~n22 & ~n23;
  assign n25 = n21 & n24;
  assign n26 = pi0 & n13;
  assign n27 = ~pi0 & ~n13;
  assign n28 = ~n26 & ~n27;
  assign n29 = n13 & ~n28;
  assign n30 = ~n13 & n28;
  assign n31 = ~n29 & ~n30;
  assign n32 = n25 & n31;
  assign n33 = ~n13 & ~n31;
  assign n34 = ~n32 & ~n33;
  assign n35 = n18 & ~n34;
  assign n36 = ~n13 & n35;
  assign n37 = ~n13 & ~n35;
  assign n38 = ~n36 & ~n37;
  assign n39 = n13 & n28;
  assign n40 = ~n13 & ~n28;
  assign n41 = ~n39 & ~n40;
  assign n42 = pi5 & n13;
  assign n43 = pi5 & ~n13;
  assign n44 = ~n42 & ~n43;
  assign n45 = pi4 & n13;
  assign n46 = pi4 & ~n13;
  assign n47 = ~n45 & ~n46;
  assign n48 = pi7 & n13;
  assign n49 = pi7 & ~n13;
  assign n50 = ~n48 & ~n49;
  assign n51 = pi6 & n13;
  assign n52 = pi6 & ~n13;
  assign n53 = ~n51 & ~n52;
  assign n54 = n21 & ~n24;
  assign n55 = n21 & ~n54;
  assign n56 = ~n41 & ~n55;
  assign n57 = ~n29 & ~n56;
  assign n58 = ~n18 & ~n57;
  assign n59 = n18 & n57;
  assign n60 = ~n58 & ~n59;
  assign n61 = ~n18 & n35;
  assign n62 = ~n35 & ~n60;
  assign n63 = ~n61 & ~n62;
  assign n64 = n41 & ~n55;
  assign n65 = ~n41 & n55;
  assign n66 = ~n64 & ~n65;
  assign n67 = ~n28 & n35;
  assign n68 = ~n35 & ~n66;
  assign n69 = ~n67 & ~n68;
  assign n70 = n35 & ~n44;
  assign n71 = ~n35 & ~n44;
  assign n72 = ~n70 & ~n71;
  assign n73 = n35 & ~n47;
  assign n74 = ~n35 & ~n47;
  assign n75 = ~n73 & ~n74;
  assign n76 = n72 & n75;
  assign n77 = ~n24 & n35;
  assign n78 = n24 & ~n35;
  assign n79 = ~n77 & ~n78;
  assign n80 = n35 & ~n79;
  assign n81 = ~n35 & n79;
  assign n82 = ~n80 & ~n81;
  assign n83 = n76 & n82;
  assign n84 = ~n35 & ~n82;
  assign n85 = ~n83 & ~n84;
  assign n86 = ~n21 & ~n24;
  assign n87 = ~n25 & ~n86;
  assign n88 = ~n21 & n35;
  assign n89 = ~n35 & ~n87;
  assign n90 = ~n88 & ~n89;
  assign n91 = n38 & ~n90;
  assign n92 = ~n38 & n90;
  assign n93 = ~n91 & ~n92;
  assign n94 = ~n85 & n93;
  assign n95 = ~n38 & ~n93;
  assign n96 = ~n94 & ~n95;
  assign n97 = n69 & ~n96;
  assign n98 = n63 & n97;
  assign n99 = ~n38 & n98;
  assign n100 = ~n38 & ~n98;
  assign n101 = ~n99 & ~n100;
  assign n102 = n38 & n90;
  assign n103 = ~n38 & ~n90;
  assign n104 = ~n102 & ~n103;
  assign n105 = n35 & n79;
  assign n106 = ~n35 & ~n79;
  assign n107 = ~n105 & ~n106;
  assign n108 = n35 & ~n50;
  assign n109 = ~n35 & ~n50;
  assign n110 = ~n108 & ~n109;
  assign n111 = n35 & ~n53;
  assign n112 = ~n35 & ~n53;
  assign n113 = ~n111 & ~n112;
  assign n114 = n72 & ~n75;
  assign n115 = n72 & ~n114;
  assign n116 = ~n107 & ~n115;
  assign n117 = ~n80 & ~n116;
  assign n118 = ~n104 & ~n117;
  assign n119 = ~n91 & ~n118;
  assign n120 = n69 & ~n119;
  assign n121 = n69 & ~n120;
  assign n122 = ~n63 & ~n121;
  assign n123 = n63 & n121;
  assign n124 = ~n122 & ~n123;
  assign n125 = ~n63 & n98;
  assign n126 = ~n98 & ~n124;
  assign n127 = ~n125 & ~n126;
  assign n128 = ~n69 & ~n119;
  assign n129 = n69 & n119;
  assign n130 = ~n128 & ~n129;
  assign n131 = ~n69 & n98;
  assign n132 = ~n98 & ~n130;
  assign n133 = ~n131 & ~n132;
  assign n134 = n104 & ~n117;
  assign n135 = ~n104 & n117;
  assign n136 = ~n134 & ~n135;
  assign n137 = ~n90 & n98;
  assign n138 = ~n98 & ~n136;
  assign n139 = ~n137 & ~n138;
  assign n140 = n98 & ~n110;
  assign n141 = ~n98 & ~n110;
  assign n142 = ~n140 & ~n141;
  assign n143 = n98 & ~n113;
  assign n144 = ~n98 & ~n113;
  assign n145 = ~n143 & ~n144;
  assign n146 = n142 & n145;
  assign n147 = ~n75 & n98;
  assign n148 = n75 & ~n98;
  assign n149 = ~n147 & ~n148;
  assign n150 = n98 & ~n149;
  assign n151 = ~n98 & n149;
  assign n152 = ~n150 & ~n151;
  assign n153 = n146 & n152;
  assign n154 = ~n98 & ~n152;
  assign n155 = ~n153 & ~n154;
  assign n156 = ~n35 & n98;
  assign n157 = ~n35 & ~n98;
  assign n158 = ~n156 & ~n157;
  assign n159 = ~n72 & ~n75;
  assign n160 = ~n76 & ~n159;
  assign n161 = ~n72 & n98;
  assign n162 = ~n98 & ~n160;
  assign n163 = ~n161 & ~n162;
  assign n164 = n158 & ~n163;
  assign n165 = ~n158 & n163;
  assign n166 = ~n164 & ~n165;
  assign n167 = ~n155 & n166;
  assign n168 = ~n158 & ~n166;
  assign n169 = ~n167 & ~n168;
  assign n170 = n107 & ~n115;
  assign n171 = ~n107 & n115;
  assign n172 = ~n170 & ~n171;
  assign n173 = ~n79 & n98;
  assign n174 = ~n98 & ~n172;
  assign n175 = ~n173 & ~n174;
  assign n176 = n101 & ~n175;
  assign n177 = ~n101 & n175;
  assign n178 = ~n176 & ~n177;
  assign n179 = ~n169 & n178;
  assign n180 = ~n101 & ~n178;
  assign n181 = ~n179 & ~n180;
  assign n182 = n139 & ~n181;
  assign n183 = n133 & n182;
  assign po3 = ~n127 | ~n183;
  assign n185 = ~n101 & ~po3;
  assign n186 = ~n101 & po3;
  assign po0 = n185 | n186;
  assign n188 = ~n158 & ~po3;
  assign n189 = ~n158 & po3;
  assign po1 = n188 | n189;
  assign n191 = ~n98 & ~po3;
  assign n192 = ~n98 & po3;
  assign po2 = n191 | n192;
endmodule


