import React, { useEffect, useState } from 'react';
import { View, Text, Button, Alert, ScrollView } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { getReferenceInfo, reloadReference, uploadReferenceVideo } from '../services/api';

export default function ReferenceScreen() {
	const [info, setInfo] = useState(null);
	const [loading, setLoading] = useState(false);

	const refresh = async () => {
		try {
			const data = await getReferenceInfo();
			setInfo(data);
		} catch (e) {
			setInfo(null);
		}
	};

	useEffect(() => {
		refresh();
	}, []);

	const onReload = async () => {
		setLoading(true);
		try {
			await reloadReference();
			await refresh();
		} catch (e) {
			Alert.alert('Error', 'Failed to reload reference');
		} finally {
			setLoading(false);
		}
	};

	const onUpload = async () => {
		const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
		if (!permission.granted) {
			Alert.alert('Permission needed', 'Please allow access to your photo library to upload videos.');
			return;
		}
		const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['videos'], quality: 0.7 });
		if (!result.canceled && result.assets?.length > 0) {
			setLoading(true);
			try {
				await uploadReferenceVideo(result.assets[0].uri);
				await refresh();
			} catch (e) {
				Alert.alert('Error', 'Failed to upload reference');
			} finally {
				setLoading(false);
			}
		}
	};

	return (
		<ScrollView contentContainerStyle={{ padding: 16 }}>
			<Text style={{ fontSize: 20, fontWeight: '700', marginBottom: 8 }}>Reference</Text>

			<Button title={loading ? 'Working...' : 'Reload default reference.MOV'} onPress={onReload} disabled={loading} />
			<View style={{ height: 8 }} />
			<Button title={loading ? 'Working...' : 'Upload new reference video'} onPress={onUpload} disabled={loading} />

			<View style={{ marginTop: 16 }}>
				<Text style={{ fontWeight: '700', marginBottom: 6 }}>Current Reference Info</Text>
				{info ? (
					<View>
						<Text>Path: {info.reference_video || 'N/A'}</Text>
						{info.swing_phases && (
							<View style={{ marginTop: 8 }}>
								<Text style={{ fontWeight: '600' }}>Swing Phases</Text>
								{Object.entries(info.swing_phases).map(([k, v]) => (
									<Text key={k}>{k}: {v}</Text>
								))}
							</View>
						)}
					</View>
				) : (
					<Text>No reference loaded.</Text>
				)}
			</View>
		</ScrollView>
	);
} 