module top (pi0, pi1, pi2, pi3, pi4, pi5, pi6, po0, po1, po2, po3);

  input pi0, pi1, pi2, pi3, pi4, pi5, pi6;
  output po0, po1, po2, po3;
  wire n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14,
    n16, n17, n18, n19, n21, n22, n23, n24;

  assign n0 = ~pi0 & pi3;
  assign n1 = pi0 & ~pi3;
  assign n2 = pi0 & pi3;
  assign n3 = ~pi1 & pi4;
  assign n4 = pi1 & ~pi4;
  assign n5 = pi1 & pi4;
  assign n6 = ~pi2 & pi5;
  assign n7 = pi2 & ~pi5;
  assign n8 = pi2 & pi5;
  assign n9 = ~n0 & ~n1;
  assign n10 = ~n3 & ~n4;
  assign n11 = ~n6 & ~n7;
  assign n12 = pi6 & n9;
  assign n13 = ~pi6 & ~n9;
  assign n14 = pi6 & ~n9;
  assign po0 = n12 | n13;
  assign n16 = ~n14 & ~n2;
  assign n17 = n10 & ~n16;
  assign n18 = ~n10 & n16;
  assign n19 = ~n10 & ~n16;
  assign po1 = n17 | n18;
  assign n21 = ~n19 & ~n5;
  assign n22 = n11 & ~n21;
  assign n23 = ~n11 & n21;
  assign n24 = ~n11 & ~n21;
  assign po2 = n22 | n23;
  assign po3 = n24 | n8;
endmodule
