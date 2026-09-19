// this file is 

// imports + explanation of use
import 'package:flutter/material.dart';

import 'dart:ui' as ui; // gives us access to canvas for drawing

// http lets us talk to the FastAPI backend
// typed_data lets us work with raw bytes (for sending image of board to backend)
import 'dart:typed_data'; 

void main() {
  // runApp starts the whole flutter engine and inflates our widget tree
  runApp(const SudoTayApp());
}

class SudoTayApp extends StatelessWidget {
  // stateless widget because the app itself doesn't change; 
  // it just defines the theme and handles routing to the drawing page
  const SudoTayApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sudo-Tay',
      theme: ThemeData.dark(),
      home: const DrawingPage(),
    );
}

class DrawingPage extends StatefulWidget {
  // stateful widget because the drawing canvas CHANGES
  // every stroke the user makes updates the state of the canvas
  const DrawingPage({super.key});

  @override
  State<DrawingPage> createState() => _DrawingPageState();
}

class _DrawingPageState extends State<DrawingPage> {
  // a list of points that the user has drawn on the canvas
  // offset is a 2D point (x,y) in the canvas coordinate system
  // null acts as a "pen up" signal, which separates the strokes
  final List<Offset?> points = [];

  
}