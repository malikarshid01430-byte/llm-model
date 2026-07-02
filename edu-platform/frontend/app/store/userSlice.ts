import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type UserState = {
  id: string | null;
  email: string | null;
  fullName: string | null;
  role: string | null;
  token: string | null;
};

const initialState: UserState = {
  id: null,
  email: null,
  fullName: null,
  role: null,
  token: null,
};

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUser(state, action: PayloadAction<UserState>) {
      return { ...state, ...action.payload };
    },
    clearUser() {
      return initialState;
    },
  },
});

export const { setUser, clearUser } = userSlice.actions;
export default userSlice.reducer;
