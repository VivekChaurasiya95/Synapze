import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { userService } from '../services/userService';
import { Trophy, Medal, Award, Star, Users } from 'lucide-react';
import { PageLoading, Avatar } from '../components';
import { formatNumber } from '../utils/helpers';

const Leaderboard = () => {
  const [loading, setLoading] = useState(true);
  const [users, setUsers] = useState([]);
  const [filter, setFilter] = useState('all');
  const [error, setError] = useState(null);

  useEffect(() => {
    loadLeaderboard();
  }, [filter]);

  const loadLeaderboard = async () => {
    try {
      setLoading(true);
      setError(null);
      const params = filter !== 'all' ? { role: filter } : {};
      const response = await userService.getLeaderboard({ ...params, limit: 50 });
      setUsers(response?.users || []);
    } catch (error) {
      console.error('Error loading leaderboard:', error);
      setError('Failed to load leaderboard. Please try again later.');
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  const getRankIcon = (rank) => {
    switch (rank) {
      case 1:
        return <Trophy className="w-6 h-6 text-yellow-500 shadow-glow" />;
      case 2:
        return <Medal className="w-6 h-6 text-slate-300" />;
      case 3:
        return <Medal className="w-6 h-6 text-amber-500" />;
      default:
        return (
          <span className="w-6 h-6 flex items-center justify-center text-brand-text-muted font-bold text-xs ring-1 ring-brand-border rounded-full">
            {rank}
          </span>
        );
    }
  };

  const getRankBgColor = (rank) => {
    switch (rank) {
      case 1:
        return 'bg-yellow-500/5 group-hover:bg-yellow-500/10';
      case 2:
        return 'bg-slate-500/5 group-hover:bg-slate-500/10';
      case 3:
        return 'bg-amber-500/5 group-hover:bg-amber-500/10';
      default:
        return '';
    }
  };

  return (
    <div className="wrapper py-8 animate-fade-in text-white">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <Trophy className="w-10 h-10 text-brand-accent" />
            <h1 className="text-4xl font-bold tracking-tight">
              Skill <span className="text-gradient">Leaderboard</span>
            </h1>
          </div>
          <p className="text-brand-text-secondary text-lg text-left">
            Celebrating the top contributors and experts in the MITS community.
          </p>
        </div>
        
        <div className="flex bg-brand-surface p-1 rounded-xl border border-brand-border h-12">
          {[
            { id: 'all', label: 'All', icon: Users },
            { id: 'student', label: 'Students', icon: Award },
            { id: 'teacher', label: 'Teachers', icon: Star },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setFilter(item.id)}
              className={`flex items-center space-x-2 px-6 rounded-lg text-xs font-bold uppercase tracking-widest transition-all ${
                filter === item.id
                  ? 'bg-brand-accent text-white shadow-glow'
                  : 'text-brand-text-muted hover:text-brand-text-secondary'
              }`}
            >
              <item.icon size={16} />
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Error State */}
      {error && !loading && (
        <div className="mb-8 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-center">
          <p>{error}</p>
          <button
            onClick={loadLeaderboard}
            className="mt-2 px-4 py-1 text-sm font-bold bg-red-500/20 hover:bg-red-500/30 rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Podium Section */}
      {!loading && users.length >= 3 && (
        <div className="podium-section flex items-end justify-center space-x-4 mb-20 px-4">
          {/* Second place */}
          <div className="text-center group flex-1 max-w-[140px]">
            <Link to={`/user/${users[1]?._id}`}>
              <div className="relative mb-4 group-hover:-translate-y-2 transition-transform duration-500">
                <div className="relative mx-auto inline-block">
                   <Avatar
                     name={users[1]?.name}
                     src={users[1]?.avatar}
                     size="large"
                     className="ring-[6px] ring-slate-500/30"
                   />
                   <div className="absolute -bottom-2 right-1/2 translate-x-1/2 w-8 h-8 bg-slate-500 rounded-full flex items-center justify-center text-white font-bold border-4 border-brand-dark overflow-hidden">
                     2
                   </div>
                </div>
              </div>
              <p className="font-bold text-white truncate px-2 mb-1">
                {users[1]?.name}
              </p>
              <p className="text-slate-400 font-bold text-sm mb-4">
                {formatNumber(users[1]?.totalPoints)} <span className="text-[10px] uppercase tracking-tighter">PTS</span>
              </p>
            </Link>
            <div className="w-full h-24 bg-gradient-to-b from-slate-500/20 to-transparent rounded-t-2xl border-t border-x border-slate-500/30 shadow-2xl" />
          </div>

          {/* First place */}
          <div className="text-center group flex-1 max-w-[170px]">
            <Link to={`/user/${users[0]?._id}`}>
              <div className="relative mb-5 group-hover:-translate-y-3 transition-transform duration-500">
                <div className="relative mx-auto inline-block">
                   <Avatar
                     name={users[0]?.name}
                     src={users[0]?.avatar}
                     size="xl"
                     className="ring-[8px] ring-yellow-500/30 shadow-[0_0_30px_rgba(234,179,8,0.2)]"
                   />
                   <div className="absolute -bottom-3 right-1/2 translate-x-1/2 w-10 h-10 bg-yellow-500 rounded-full flex items-center justify-center text-white shadow-glow border-4 border-brand-dark overflow-hidden">
                     <Trophy size={18} />
                   </div>
                </div>
              </div>
              <p className="font-bold text-white text-xl truncate px-2 mb-1">
                {users[0]?.name}
              </p>
              <p className="text-yellow-500 font-bold text-lg mb-6">
                {formatNumber(users[0]?.totalPoints)} <span className="text-xs uppercase tracking-tighter">PTS</span>
              </p>
            </Link>
            <div className="w-full h-40 bg-gradient-to-b from-yellow-500/30 to-transparent rounded-t-2xl border-t border-x border-yellow-500/40 shadow-[0_0_40px_rgba(234,179,8,0.1)]" />
          </div>

          {/* Third place */}
          <div className="text-center group flex-1 max-w-[140px]">
            <Link to={`/user/${users[2]?._id}`}>
              <div className="relative mb-4 group-hover:-translate-y-2 transition-transform duration-500">
                <div className="relative mx-auto inline-block">
                   <Avatar
                     name={users[2]?.name}
                     src={users[2]?.avatar}
                     size="large"
                     className="ring-[6px] ring-amber-700/30"
                   />
                   <div className="absolute -bottom-2 right-1/2 translate-x-1/2 w-8 h-8 bg-amber-700 rounded-full flex items-center justify-center text-white font-bold border-4 border-brand-dark overflow-hidden">
                     3
                   </div>
                </div>
              </div>
              <p className="font-bold text-white truncate px-2 mb-1">
                {users[2]?.name}
              </p>
              <p className="text-amber-600 font-bold text-sm mb-4">
                {formatNumber(users[2]?.totalPoints)} <span className="text-[10px] uppercase tracking-tighter">PTS</span>
              </p>
            </Link>
            <div className="w-full h-16 bg-gradient-to-b from-amber-700/20 to-transparent rounded-t-2xl border-t border-x border-amber-700/30 shadow-2xl" />
          </div>
        </div>
      )}

      {/* Full List */}
      {loading ? (
        <div className="space-y-4">
           {[1,2,3,4,5].map(i => <div key={i} className="h-16 w-full card animate-pulse bg-brand-surface/50" />)}
        </div>
      ) : (
        <div className="card shadow-2xl border-brand-border/30 overflow-hidden bg-brand-surface/10 p-0">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-brand-surface/50 border-b border-brand-border text-left">
                  <th className="px-8 py-5 text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text-muted">
                    Rank
                  </th>
                  <th className="px-8 py-5 text-left text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text-muted">
                    Contributor
                  </th>
                  <th className="px-8 py-5 text-center text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text-muted">
                    Role
                  </th>
                  <th className="px-8 py-5 text-center text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text-muted">
                    Credits
                  </th>
                  <th className="px-8 py-5 text-center text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text-muted">
                    Exp
                  </th>
                  <th className="px-8 py-5 text-right text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text-muted">
                    Total Points
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-brand-border/30">
                {users.map((leaderUser) => (
                  <tr
                    key={leaderUser._id}
                    className={`group transition-all hover:bg-white/5 ${getRankBgColor(
                      leaderUser.rank
                    )}`}
                  >
                    <td className="px-8 py-5 font-bold">
                      <div className="flex items-center">
                        {getRankIcon(leaderUser.rank)}
                      </div>
                    </td>
                    <td className="px-8 py-5 text-left">
                      <Link
                        to={`/user/${leaderUser._id}`}
                        className="flex items-center space-x-4 hover:translate-x-1 transition-transform"
                      >
                        <Avatar name={leaderUser.name} src={leaderUser.avatar} className="border border-white/5" />
                        <span className="font-bold text-white group-hover:text-brand-accent transition-colors">
                          {leaderUser.name}
                        </span>
                      </Link>
                    </td>
                    <td className="px-8 py-5 text-center">
                      <span
                        className={`text-[10px] font-bold uppercase border px-2 py-1 rounded-md ${
                          leaderUser.role === 'teacher'
                            ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                            : 'bg-brand-accent/10 text-brand-accent border-brand-accent/20'
                        }`}
                      >
                        {leaderUser.role}
                      </span>
                    </td>
                    <td className="px-8 py-5 text-center">
                      <span className="font-medium text-brand-text-secondary">
                        {leaderUser.tasksCompleted || 0}
                      </span>
                    </td>
                    <td className="px-8 py-5 text-center">
                      <div className="flex items-center justify-center space-x-2">
                        <Star className="w-4 h-4 text-yellow-500 fill-yellow-500/20" />
                        <span className="font-bold text-brand-text-secondary text-sm">
                          {leaderUser.averageRating?.toFixed(1) || '0.0'}
                        </span>
                      </div>
                    </td>
                    <td className="px-8 py-5 text-right">
                      <span className="font-black text-brand-accent text-lg">
                        {formatNumber(leaderUser.totalPoints)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Leaderboard;
