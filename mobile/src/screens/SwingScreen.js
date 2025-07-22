import React, { useState } from 'react';
import { View, Text, Button, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { analyzeSwing } from '../services/api';

export default function SwingScreen() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const pickVideo = async () => {
        console.log('Upload button pressed');

        const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!permission.granted) {
            Alert.alert('Permission needed', 'Please allow access to your photo library to upload videos.');
            return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['videos'],
            quality: 0.7,
        });

        console.log('Picker response:', result);

        if (!result.canceled && result.assets?.length > 0) {
            setLoading(true);
            try {
                const analysisResult = await analyzeSwing(result.assets[0].uri);
                setResult(analysisResult);
            } catch (error) {
                Alert.alert('Error', 'Analysis failed');
            } finally {
                setLoading(false);
            }
        }      
    };
    
    return (
        <View style={{ flex: 1, padding: 20, justifyContent: 'center' }}>
            <Text style={{ fontSize: 20, textAlign: 'center', marginBottom: 30 }}>
                Swingalyze
            </Text>

            <Button title="Upload Video" onPress={pickVideo} />

            {loading && <Text>Analyzing...</Text>}

            {result && (
                <View style={{ marginTop: 20}}>
                    <Text>Score: {result.score}%</Text>
                    <Text>Feedback:</Text>
                    {result.feedback.map((item, index) => (
                        <Text key={index}>• {item}</Text>
                    ))}
                </View>
            )}
        </View>
    )
}
