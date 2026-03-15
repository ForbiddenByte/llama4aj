package com.example.desktopapp;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import com.llama4aj;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main extends Application {

    private TextArea chatArea;
    private TextField promptInput;
    private Button sendButton;
    private Button stopButton;

    private llama4aj model;
    private ExecutorService executorService;
    private StringBuilder currentResponse;
    private volatile boolean isGenerating = false;

    // Placeholder for model path - user needs to set this
    private static final String MODEL_PATH = "./model.gguf"; // !!! IMPORTANT: CHANGE THIS !!!

    @Override
    public void start(Stage primaryStage) {
        chatArea = new TextArea();
        chatArea.setEditable(false);
        chatArea.setWrapText(true);
        VBox.setVgrow(chatArea, Priority.ALWAYS);

        promptInput = new TextField();
        promptInput.setPromptText("Enter your prompt...");
        HBox.setHgrow(promptInput, Priority.ALWAYS);

        sendButton = new Button("Send");
        sendButton.setDefaultButton(true);
        sendButton.setOnAction(e -> sendMessage());

        stopButton = new Button("Stop");
        stopButton.setDisable(true);
        stopButton.setOnAction(e -> interruptGeneration());

        HBox inputRow = new HBox(8, promptInput, sendButton, stopButton);

        VBox root = new VBox(10, chatArea, inputRow);
        root.setStyle("-fx-padding: 10;");

        Scene scene = new Scene(root, 700, 500);

        primaryStage.setTitle("Llama Desktop App");
        primaryStage.setScene(scene);
        primaryStage.show();

        executorService = Executors.newSingleThreadExecutor();

        // Initialize model in a background thread
        executorService.submit(this::initializeModel);
    }

    private void initializeModel() {
        Platform.runLater(() -> chatArea.appendText("Loading model...\n"));

        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            Platform.runLater(() -> {
                chatArea.appendText("ERROR: Model file not found at " + MODEL_PATH + "\n");
                sendButton.setDisable(true);
            });
            return;
        }

        try {
            model = llama4aj.load(MODEL_PATH);
            if (model != null) {
                Platform.runLater(() -> {
                    chatArea.appendText("Model loaded successfully!\n");
                    sendButton.setDisable(false);
                });
            } else {
                Platform.runLater(() -> {
                    chatArea.appendText("ERROR: Failed to load model.\n");
                    sendButton.setDisable(true);
                });
            }
        } catch (Exception e) {
            Platform.runLater(() -> {
                chatArea.appendText("ERROR during model loading: " + e.getMessage() + "\n");
                sendButton.setDisable(true);
            });
        }
    }

    private void sendMessage() {
        String prompt = promptInput.getText().trim();
        if (prompt.isEmpty() || isGenerating) {
            return;
        }

        chatArea.appendText("You: " + prompt + "\n");
        promptInput.clear();
        setGeneratingState(true);

        executorService.submit(() -> generateResponse(prompt));
    }

    private void generateResponse(String userPrompt) {
        if (model == null) {
            Platform.runLater(() -> chatArea.appendText("ERROR: Model not loaded.\n"));
            setGeneratingState(false);
            return;
        }

        // FIX 1: Reset response buffer before every new generation
        currentResponse = new StringBuilder();

        try {
            String fullPrompt = "User: " + userPrompt + "\nAssistant:";

            // FIX 2: Print the label once upfront, before streaming begins
            Platform.runLater(() -> chatArea.appendText("Assistant: "));

            model.generate(
                    new llama4aj.CompletionParams()
                            .prompt(fullPrompt)
                            .nPredict(256)
                            .stream(true),

                    // FIX 3: Append only the new token — not the full accumulated response
                    token -> {
                        currentResponse.append(token);
                        Platform.runLater(() -> chatArea.appendText(token));
                    },

                    // On completion: newline + re-enable UI
                    () -> Platform.runLater(() -> {
                        chatArea.appendText("\n");
                        setGeneratingState(false);
                    }));

        } catch (Exception e) {
            Platform.runLater(() -> chatArea.appendText("ERROR during generation: " + e.getMessage() + "\n"));
            setGeneratingState(false);
        }
    }

    private void interruptGeneration() {
        if (model != null) {
            // Run interrupt on the executor so it's serialized with generation
            executorService.submit(() -> model.interrupt());
        }
    }

    private void setGeneratingState(boolean generating) {
        isGenerating = generating;
        Platform.runLater(() -> {
            sendButton.setDisable(generating);
            stopButton.setDisable(!generating);
            promptInput.setDisable(generating);
        });
    }

    @Override
    public void stop() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdownNow();
        }
        if (model != null) {
            model.close();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}