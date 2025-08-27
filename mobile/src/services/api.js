import axios from 'axios';

const API_BASE_URL = 'http://192.168.68.72:8000'; // change later for production

const guessMimeFromUri = (uri) => {
	try {
		const cleanUri = uri.split('?')[0];
		const ext = cleanUri.substring(cleanUri.lastIndexOf('.') + 1).toLowerCase();
		switch (ext) {
			case 'mov':
				return 'video/quicktime';
			case 'mp4':
				return 'video/mp4';
			case 'mkv':
				return 'video/x-matroska';
			case 'avi':
				return 'video/x-msvideo';
			default:
				return 'video/mp4';
		}
	} catch (e) {
		return 'video/mp4';
	}
};

const guessNameFromUri = (uri) => {
	try {
		const cleanUri = uri.split('?')[0];
		const ext = cleanUri.substring(cleanUri.lastIndexOf('.') + 1).toLowerCase();
		return `swing.${ext || 'mp4'}`;
	} catch (e) {
		return 'swing.mp4';
	}
};

// videoUri is file path pointing to selected vid
export const analyzeSwing = async (videoUri) => {
	const formData = new FormData();
	const mime = guessMimeFromUri(videoUri);
	const name = guessNameFromUri(videoUri);
	formData.append('file', {
		uri: videoUri,
		type: mime,
		name,
	});

	try {
		const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
			headers: {
				'Content-Type': 'multipart/form-data',
			},
		});
		return response.data;
	} catch (error) {
		console.error('Analysis failed:', error?.response?.data || error.message);
		throw error;
	}
};

export const getReferenceInfo = async () => {
	const res = await axios.get(`${API_BASE_URL}/reference-info`);
	return res.data;
};

export const reloadReference = async () => {
	const res = await axios.post(`${API_BASE_URL}/reload-reference`);
	return res.data;
};

export const uploadReferenceVideo = async (videoUri) => {
	const formData = new FormData();
	const mime = guessMimeFromUri(videoUri);
	const name = guessNameFromUri(videoUri).replace('swing', 'reference');
	formData.append('file', {
		uri: videoUri,
		type: mime,
		name,
	});
	const res = await axios.post(`${API_BASE_URL}/upload-reference`, formData, {
		headers: { 'Content-Type': 'multipart/form-data' },
	});
	return res.data;
};